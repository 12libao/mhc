import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver
from rich import box
from rich.console import Console
from rich.table import Table


def profiler_wrapper(fn, *args, **kwargs):
    result_path = "./result_profiling"
    skip_first = 10
    wait = 0
    warmup = 10
    active = 10
    repeat = 1
    stream = torch.npu.current_stream()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    
    # 先跑一次触发 triton 编译，避免编译开销污染 profiling
    fn(*args, **kwargs)
    stream.synchronize()
    
    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat,
                                                 skip_first=skip_first),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_path),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config) as prof:
        stream.synchronize()
        for i in range(skip_first + (wait + warmup + active) * repeat):
            fn(*args, **kwargs)
            prof.step()
        stream.synchronize()


def rel_err(a, b):
  return (torch.norm((a - b).abs()) / (torch.norm(b) + 1e-12)).item()


def bench(fn, *args, iters=10, warmup=5):
    # 1. Warmup
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()

    # 2. 设置 Memory 监测
    torch.npu.reset_peak_memory_stats()
    begin_mem = torch.npu.memory_allocated()
    
    # 3. 计时
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    
    with torch.no_grad():
        start_event.record()
        for _ in range(iters):
            out = fn(*args)
        end_event.record()
    torch.npu.synchronize()
    
    # 4. 计算指标
    elapsed_time_us = start_event.elapsed_time(end_event) / iters * 1000  # us
    peak_mem = torch.npu.max_memory_allocated()
    active_mem_mb = (peak_mem - begin_mem) / 1024 / 1024  # MB

    return (*out, elapsed_time_us, active_mem_mb)

def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


# -----------------------------
# 参考实现（你的 PyTorch 源码）
# -----------------------------
def hc_pre_reference(x, hc_weight, alpha_scales, bias, norm_eps=1e-6, hc_eps=1e-6):
    B, S, N, D = x.shape
    X_flat = x.view(B, S, -1).float()
    rsqrt = torch.rsqrt(X_flat.square().mean(-1, keepdim=True) + norm_eps)

    X_hat = torch.matmul(X_flat, hc_weight)  # FP32
    X_hat = X_hat * rsqrt

    splits = [N, N, N * N]
    X_pre, X_post, X_comb = torch.split(X_hat, splits, dim=-1)

    H_pre = torch.sigmoid(alpha_scales[0] * X_pre + bias[0]) + hc_eps  # FP32
    x_fp32 = x.float()
    weighted_X = H_pre.unsqueeze(-1) * x_fp32
    h_in = weighted_X.sum(dim=2).to(torch.bfloat16)

    H_post = 2.0 * torch.sigmoid(alpha_scales[1] * X_post + bias[1])  # FP32
    X_comb = X_comb.view(B, S, N, N)
    H_comb_before = alpha_scales[2] * X_comb + bias[2]  # FP32
    return h_in, H_post, H_comb_before


# -----------------------------------------
# Kernel 1: 计算每个 (B,S) 行的 rsqrt
# 输入 x_flat: [M, K] BF16，输出 rsqrt: [M] FP32
# -----------------------------------------
@triton.jit
def rms_rsqrt_kernel(
    x_flat_ptr,          # [M, K]
    rsqrt_ptr,           # [M]
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    norm_eps: tl.constexpr,
):
    pid_m = tl.program_id(0)              # 0..ceil_div(M, BLOCK_M)-1
    m_start = pid_m * BLOCK_M

    offs_m = m_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # 用 block_ptr 方便沿 K 方向 advance
    x_blk_ptr = tl.make_block_ptr(
        base=x_flat_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(m_start, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # kernel 内部切 K：每次处理 (BLOCK_M, BLOCK_K) 的 tile
    for _ in tl.static_range(0, K, BLOCK_K):
        x_blk = tl.load(x_blk_ptr,boundary_check=(0, 1),padding_option="zero").to(tl.float32)
        acc += tl.sum(x_blk * x_blk, axis=1)
        x_blk_ptr = tl.advance(x_blk_ptr, (0, BLOCK_K))

    mean = acc / K
    r = tl.rsqrt(mean + norm_eps)
    tl.store(rsqrt_ptr + offs_m, r, mask=m_mask)

# ----------------------------------------------------------
# Kernel 2: MatMul + scale(rsqrt) + split + 写出：
#   H_pre_tmp: [M, N] FP32
#   H_post   : [M, N] FP32
#   H_comb   : [M, N*N] FP32 (后续 reshape 为 [B,S,N,N])
# ----------------------------------------------------------
# @triton.autotune(
#     configs=[triton.Config({"BLOCK_M": 32, "BLOCK_K": 32})],
#     key=["M", "NT", "K"],
# )
@triton.jit
def hc_matmul_post_kernel(
    x_ptr, w_ptr,
    rsqrt_ptr,
    hpre_ptr,  # [M,N] fp32
    hpost_ptr, # [M,N] fp32
    hcomb_ptr, # [M,N*N] fp32
    bias0_ptr, bias1_ptr, bias2_ptr,
    M: tl.constexpr, NT: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    alpha0: tl.constexpr, alpha1: tl.constexpr, alpha2: tl.constexpr,
    hc_eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)          # 0 .. grid-1
    m_start = pid * BLOCK_M         # 该 program 负责的 M 起始行
    m = m_start + tl.arange(0, BLOCK_M)  # [BM]

    # 加载 bias（要求 N、N*N 为 tl.constexpr，且 bias 存储连续）
    bias0 = tl.load(bias0_ptr + tl.arange(0, N))
    bias1 = tl.load(bias1_ptr + tl.arange(0, N))
    bias2 = tl.load(bias2_ptr + tl.arange(0, N * N))

    # -----------------------------------------------------------
    # 1) MatMul
    # -----------------------------------------------------------
    x_blk_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(m_start, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_blk_ptr = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, NT),
        strides=(NT, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_K, NT),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, NT), dtype=tl.float32)
    acc_rsqrt = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        x_blk = tl.load(x_blk_ptr, boundary_check=(0, 1), padding_option="zero")
        w_blk = tl.load(w_blk_ptr, boundary_check=(0, 1), padding_option="zero")

        x_blk = x_blk.to(tl.float32)
        acc_rsqrt += tl.sum(x_blk * x_blk, axis=1)
        acc = tl.dot(x_blk, w_blk, acc)

        x_blk_ptr = tl.advance(x_blk_ptr, (0, BLOCK_K))
        w_blk_ptr = tl.advance(w_blk_ptr, (BLOCK_K, 0))

    mean = acc_rsqrt / K
    acc_rsqrt = tl.rsqrt(mean + 1e-6)
    acc = acc * acc_rsqrt[:, None]
 
    # 写回每行的 rsqrt（原代码传了 rsqrt_ptr 但没用）
    tl.store(rsqrt_ptr + m, acc_rsqrt, mask=(m < M))

    # # load rsqrt 并 scale
    # rsqrt = tl.load(rsqrt_ptr + m, mask=(m < M))
    # acc = acc * rsqrt[:, None]

    # -----------------------------------------------------------
    # 2) Split & Store
    #    Pre:  [0, N)
    #    Post: [N, 2N)
    #    Comb: [2N, 2N+N*N)
    # -----------------------------------------------------------
    acc_pre = tl.extract_slice(acc, (0, 0), (BLOCK_M, N), (1, 1))
    acc_post = tl.extract_slice(acc, (0, N), (BLOCK_M, N), (1, 1))
    acc_comb = tl.extract_slice(acc, (0, 2 * N), (BLOCK_M, N * N), (1, 1))

    hpre = tl.sigmoid(alpha0 * acc_pre + bias0[None, :]) + hc_eps
    hpost = 2.0 * tl.sigmoid(alpha1 * acc_post + bias1[None, :])
    hcomb = alpha2 * acc_comb + bias2[None, :]

    hpre_out_ptr = tl.make_block_ptr(
        base=hpre_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(m_start, 0),
        block_shape=(BLOCK_M, N),
        order=(1, 0),
    )
    tl.store(hpre_out_ptr, hpre, boundary_check=(0, 1))

    hpost_out_ptr = tl.make_block_ptr(
        base=hpost_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(m_start, 0),
        block_shape=(BLOCK_M, N),
        order=(1, 0),
    )
    tl.store(hpost_out_ptr, hpost, boundary_check=(0, 1))

    hcomb_out_ptr = tl.make_block_ptr(
        base=hcomb_ptr,
        shape=(M, N * N),
        strides=(N * N, 1),
        offsets=(m_start, 0),
        block_shape=(BLOCK_M, N * N),
        order=(1, 0),
    )
    tl.store(hcomb_out_ptr, hcomb, boundary_check=(0, 1))
    
# ----------------------------------------------------------
# Kernel 3: h_in = sum_n (H_pre_tmp[n] * x[n,:]) over N
# 输出 [M, D] BF16
# ----------------------------------------------------------
@triton.jit
def h_in_kernel(
    x_ptr,          # *bf16, [M, 4, D]
    hpre_ptr,       # *fp32, [M, 4]
    out_ptr,        # *bf16, [M, D]
    M: tl.constexpr,
    D: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_hm: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    m_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BM]
    d_ids = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)   # [BD]
    n_ids = tl.arange(0, 4)                           # N=4

    m_mask = m_ids < M
    d_mask = d_ids < D

    # h: [BM, 4] fp32
    h = tl.load(
        hpre_ptr + m_ids[:, None] * stride_hm + n_ids[None, :] * stride_hn,
        mask=m_mask[:, None],
        other=0.0
    ).to(tl.float32)

    # x: [BM, 4, BD] bf16 -> fp32
    x = tl.load(
        x_ptr
        + m_ids[:, None, None] * stride_xm
        + n_ids[None, :, None] * stride_xn
        + d_ids[None, None, :] * stride_xd,
        mask=m_mask[:, None, None] & d_mask[None, None, :],
        other=0.0
    ).to(tl.float32)

    # acc: [BM, BD]
    acc = tl.sum(x * h[:, :, None], axis=1)

    tl.store(
        out_ptr + m_ids[:, None] * stride_om + d_ids[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & d_mask[None, :]
    )

    
# -----------------------------
# Triton-Ascend 封装函数
# -----------------------------
def triton_hc_pre(x, hc_weight, alpha_scales, bias, norm_eps=1e-6, hc_eps=1e-6):
    assert x.is_npu and hc_weight.is_npu
    x = x.contiguous()
    hc_weight = hc_weight.contiguous()

    B, S, N, D = x.shape
    M = B * S
    K = N * D
    NT = N * (N + 2)  # N + N + N*N

    # flatten/view
    x_flat_MND = x.view(M, N, D)
    x_flat_MK = x.view(M, K)

    # biases/scales
    # alpha: 标量
    a0 = 0.8
    a1 = 0.5
    a2 = 0.2
    
    bias0 = bias[0].contiguous()
    bias1 = bias[1].contiguous()
    bias2 = bias[2].contiguous().view(-1)

    # alloc outputs
    rsqrt = torch.empty((M,), device="npu:0", dtype=torch.float32)
    hpre_tmp = torch.empty((M, N), device="npu:0", dtype=torch.float32)
    hpost = torch.empty((M, N), device="npu:0", dtype=torch.float32)
    hcomb_flat = torch.empty((M, N * N), device="npu:0", dtype=torch.float32)
    h_in = torch.empty((M, D), device="npu:0", dtype=torch.bfloat16)
    
    # # # 不切 K 维度，因为要算整行的平方和
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), )
    # rms_rsqrt_kernel[grid](
    #     x, rsqrt,
    #     M=M, K=K,
    #     BLOCK_M=16, 
    #     BLOCK_K=1024,
    #     norm_eps=norm_eps,
    # )
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), )

    hc_matmul_post_kernel[grid](
        x_flat_MK, hc_weight,
        rsqrt,
        hpre_tmp, hpost, hcomb_flat,
        bias0, bias1, bias2,
        M=M, NT=NT, K=K, N=N,
        alpha0=a0, alpha1=a1, alpha2=a2,
        hc_eps=hc_eps,
        BLOCK_M=64, BLOCK_K=128,
    )

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(D, META['BLOCK_D']))
    h_in_kernel[grid](
        x_flat_MND, hpre_tmp, h_in,
        M, D,
        x_flat_MND.stride(0), x_flat_MND.stride(1), x_flat_MND.stride(2),
        hpre_tmp.stride(0), hpre_tmp.stride(1),
        h_in.stride(0), h_in.stride(1),
        BLOCK_M=32, BLOCK_D=128,  # 因为 N=4, 实际 BLOCK_M = 32*4=128
    )

    # reshape to original
    h_in = h_in.view(B, S, D)
    hpost = hpost.view(B, S, N)
    hcomb = hcomb_flat.view(B, S, N, N)

    return h_in, hpost, hcomb


# -----------------------------
# 精度验证
# -----------------------------
def assert_close_like_example(result, golden, name, atol=2**-6, rtol=2**-6):
    # 仿你给的示例：小值用 atol，大值用 rtol
    mask = golden.abs() < 1.0
    try:
        torch.testing.assert_close(result[mask], golden[mask], atol=atol, rtol=0)
        torch.testing.assert_close(result[~mask], golden[~mask], atol=0, rtol=rtol)
        print(f"{name}: OK")
    except Exception as e:
        max_abs = (result - golden).abs().max().item()
        denom = golden.abs().clamp_min(1e-6)
        max_rel = ((result - golden).abs() / denom).max().item()
        print(f"{name}: FAIL  max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        raise

def generate_test_data(B, S, N, D, device):
    
    x = torch.randn((B, S, N, D), device=device, dtype=torch.bfloat16)
    hc_weight = torch.randn((N * D, N * (N + 2)), device=device, dtype=torch.float32)

    alpha_scales = torch.tensor(
        [0.8, 0.5, 0.2], device=device, dtype=torch.float32
    )
    bias = [
        torch.randn((N,), device=device, dtype=torch.float32),
        torch.randn((N,), device=device, dtype=torch.float32),
        torch.randn((N, N), device=device, dtype=torch.float32),
    ]
    return x, hc_weight, alpha_scales, bias


def check_accuracy(x, hc_weight, alpha_scales, bias):
    
    stream = torch.npu.current_stream()
    stream.synchronize()
    h_in_t, hpost_t, hcomb_t = triton_hc_pre(x, hc_weight, alpha_scales, bias)
    h_in_g, hpost_g, hcomb_g = hc_pre_reference(x, hc_weight, alpha_scales, bias)
    stream.synchronize()
    
    # 对齐 dtype（h_in 是 bf16；其它 fp32）
    assert h_in_t.dtype == torch.bfloat16
    assert hpost_t.dtype == torch.float32
    assert hcomb_t.dtype == torch.float32
    
    assert_close_like_example(h_in_t.float(), h_in_g.float(), "h_in(bf16->fp32)", atol=2**-5, rtol=2**-5)
    assert_close_like_example(hpost_t, hpost_g, "H_post(fp32)", atol=2**-6, rtol=2**-6)
    assert_close_like_example(hcomb_t, hcomb_g, "H_comb_before(fp32)", atol=2**-6, rtol=2**-6)
    
    print("All checks passed.")
    
    return

def run_profiler(x, hc_weight, alpha_scales, bias):
    
    print("Profiling Triton mHC-Pre...")
    profiler_wrapper(triton_hc_pre, x, hc_weight, alpha_scales, bias)
    
    # print("Profiling Golden mHC-Pre...")
    # profiler_wrapper(hc_pre_reference, x, hc_weight, alpha_scales, bias)
    
    return

def run_benchmark(x, hc_weight, alpha_scales, bias):
   
    h_in_t, hpost_t, hcomb_t, t_triton, m_triton = bench(triton_hc_pre, x, hc_weight, alpha_scales, bias)
    h_in_g, hpost_g, hcomb_g, t_golden, m_golden = bench(hc_pre_reference, x, hc_weight, alpha_scales, bias)
    
    e_triton_O = rel_err(h_in_t.float(), h_in_g.float())
    
    # Display Results in Table
    console = Console()
    table = Table(title="\nmHC-Pre: B:{}, S:{}, N:{}, D:{}".format(*x.shape), box=box.SIMPLE_HEAVY)
    table.add_column("Method", justify="left", style="cyan", no_wrap=True)
    table.add_column("Time (us)", justify="right", style="magenta")
    table.add_column("Memory (MB)", justify="right", style="green")
    table.add_column("RelErr", justify="right", style="red")

    table.add_row("Golden mHC-Pre", f"{t_golden:.0f}", f"{m_golden:.1f}", "-")
    table.add_row("Triton mHC-Pre", f"{t_triton:.0f}", f"{m_triton:.1f}", f"{e_triton_O:.2e}")
    console.print(table)
    
    return


if __name__ == "__main__":
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    
    # Profiling 数据（2, 4096, 4, 1536）
    # Golden mHC-Pre: 1700 us dynamic, 1100 us static
    
    # Triton mHC-Pre: 
    # hc_matmul_post_kernel: 430 us
    # h_in_kernel: 130 us
    # total: 570 us
    
    B, S, N, D = 2, 4096, 4, 6144 # D: 1536, 2048, 3072, 6144
    x, hc_weight, alpha_scales, bias = generate_test_data(B, S, N, D, device)

    # check_accuracy(x, hc_weight, alpha_scales, bias)
    # run_profiler(x, hc_weight, alpha_scales, bias)
    run_benchmark(x, hc_weight, alpha_scales, bias)
    