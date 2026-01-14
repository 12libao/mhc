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
    elapsed_time_ms = start_event.elapsed_time(end_event) / iters
    peak_mem = torch.npu.max_memory_allocated()
    active_mem_mb = (peak_mem - begin_mem) / 1024 / 1024
    
    print(f"Finished benchmarking {fn.__name__}")

    return (*out, elapsed_time_ms, active_mem_mb)

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
    x_flat_ptr,
    rsqrt_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    norm_eps: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_m = triton.cdiv(M, BLOCK_M)

    for block_m in range(pid, num_blocks_m, num_cores):
        m_start = block_m * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M

        x_blk_ptr = tl.make_block_ptr(
            base=x_flat_ptr,
            shape=(M, K),
            strides=(K, 1),
            offsets=(m_start, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # 沿 K 循环累加 sum(x^2)
        for k0 in range(0, K, BLOCK_K):
            x_blk = tl.load(x_blk_ptr, boundary_check=(0, 1)).to(tl.float32)
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
#
# 要点：
# - grid=(num_cores,)
# - kernel 内部 block loop
# - tl.make_block_ptr + tl.advance
# - 大 cube: tl.dot 得到 accumulator (BLOCK_M,BLOCK_N)
# - 小 vector/子块: tl.extract_slice 分半块做 sigmoid/scale/store
# ----------------------------------------------------------
# @triton.autotune(
#     configs=[triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32})],
#     key=["M", "NT", "K"],
# )
@triton.jit
def hc_matmul_post_kernel(
    x_ptr, w_ptr,
    rsqrt_ptr,
    hpre_ptr, hpost_ptr, hcomb_ptr,
    bias0_ptr, bias1_ptr, bias2_ptr,
    M: tl.constexpr, NT: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    num_cores: tl.constexpr,
    alpha0: tl.constexpr, alpha1: tl.constexpr, alpha2: tl.constexpr,
    hc_eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_m = triton.cdiv(M, BLOCK_M)

    SUB_BM: tl.constexpr = BLOCK_M // 1  # 分割成 1 个子块处理

    for block_m in range(pid, num_blocks_m, num_cores):
        m_start = block_m * BLOCK_M  # 这里只有 M 维度的 grid

        # -----------------------------------------------------------
        # 1. MatMul 计算 (不变)
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
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # acc_rsqrt = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            x_blk = tl.load(x_blk_ptr, boundary_check=(0, 1), padding_option="zero")
            w_blk = tl.load(w_blk_ptr, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(x_blk, w_blk, acc)
            # acc_rsqrt += tl.sum(x_blk * x_blk, axis=1)
            x_blk_ptr = tl.advance(x_blk_ptr, (0, BLOCK_K))
            w_blk_ptr = tl.advance(w_blk_ptr, (BLOCK_K, 0))
            
        # acc_rsqrt = tl.zeros((BLOCK_M,), dtype=tl.float32)
        # for _ in range(0, K, BLOCK_K):
        #     x_blk = tl.load(x_blk_ptr, boundary_check=(0, 1)).to(tl.float32)
        #     acc_rsqrt += tl.sum(x_blk * x_blk, axis=1)
        #     x_blk_ptr = tl.advance(x_blk_ptr, (0, BLOCK_K))

        # Row scale (rsqrt)
        # mean = acc_rsqrt / K
        # r = tl.rsqrt(mean + 1e-6)
        # acc = acc * r[:, None]
        
        # # Row scale (rsqrt)
        offs_m = m_start + tl.arange(0, BLOCK_M)
        r = tl.load(rsqrt_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        acc = acc * r[:, None]

        # -----------------------------------------------------------
        # 2. Split & Store
        # -----------------------------------------------------------
        # 对于 N=4, BLOCK_N=32:
        # Pre:  cols [0, 4)
        # Post: cols [4, 8)
        # Comb: cols [8, 24)
        
        # 辅助 range，用于加载 bias
        offs_n_N = tl.arange(0, N)

        for s in tl.parallel(BLOCK_M // SUB_BM):
            # 提取当前 Sub-Block (SUB_BM, BLOCK_N)
            # 注意：acc 依然保留在寄存器中
            acc_sub = tl.extract_slice(acc, (s * SUB_BM, 0), (SUB_BM, BLOCK_N), (1, 1))
            
            # --- 1. 处理 H_pre [Col 0:N] ---
            # 直接切片提取 (SUB_BM, N)
            acc_pre = tl.extract_slice(acc_sub, (0, 0), (SUB_BM, N), (1, 1))
            
            bias0 = tl.load(bias0_ptr + offs_n_N) # 加载 4 个 bias
            hpre = tl.sigmoid(alpha0 * acc_pre + bias0[None, :]) + hc_eps
            
            # 使用 block_ptr 存储，规避指针别名
            hpre_out_ptr = tl.make_block_ptr(
                base=hpre_ptr,
                shape=(M, N),
                strides=(N, 1),
                offsets=(m_start + s * SUB_BM, 0),
                block_shape=(SUB_BM, N),
                order=(1, 0)
            )
            # boundary_check 只需检查 M 维度 (dim 0)，N 维度是齐的
            tl.store(hpre_out_ptr, hpre, boundary_check=(0, 1))

            # --- 2. 处理 H_post [Col N:2N] ---
            # 直接切片提取 (SUB_BM, N) -> 对应 acc 的第 4~7 列
            acc_post = tl.extract_slice(acc_sub, (0, N), (SUB_BM, N), (1, 1))
            
            bias1 = tl.load(bias1_ptr + offs_n_N)
            hpost = 2.0 * tl.sigmoid(alpha1 * acc_post + bias1[None, :])
            
            hpost_out_ptr = tl.make_block_ptr(
                base=hpost_ptr,
                shape=(M, N),
                strides=(N, 1),
                offsets=(m_start + s * SUB_BM, 0),
                block_shape=(SUB_BM, N),
                order=(1, 0)
            )
            tl.store(hpost_out_ptr, hpost, boundary_check=(0, 1))

            # --- 3. 处理 H_comb [Col 2N: 2N+N*N] ---
            # 对应 acc 的第 8~23 列 (共16列)
            acc_comb = tl.extract_slice(acc_sub, (0, 2 * N), (SUB_BM, N * N), (1, 1))
            
            cols16 = tl.arange(0, N * N)
            bias2_16 = tl.load(bias2_ptr + cols16)
            hcomb_16 = alpha2 * acc_comb + bias2_16[None, :]

            hcomb_out_ptr = tl.make_block_ptr(
                base=hcomb_ptr,
                shape=(M, N * N),
                strides=(N * N, 1),
                offsets=(m_start + s * SUB_BM, 0),
                block_shape=(SUB_BM, N * N),
                order=(1, 0),
            )
            tl.store(hcomb_out_ptr, hcomb_16, boundary_check=(0, 1))
            
# ----------------------------------------------------------
# Kernel 3: h_in = sum_n (H_pre_tmp[n] * x[n,:]) over N
# 输出 [M, D] BF16
# ----------------------------------------------------------
@triton.jit
def h_in_kernel(
    x_ptr,          # x_3d [M,N,D] bf16
    hpre_ptr,       # [M,N] fp32
    out_ptr,        # [M,D] bf16
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xd: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_d = triton.cdiv(D, BLOCK_D)
    num_blocks = M * num_blocks_d

    for block_idx in range(pid, num_blocks, num_cores):
        m = block_idx // num_blocks_d
        bd = block_idx % num_blocks_d
        d_start = bd * BLOCK_D

        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # block pointers: x_row:[N,D], hpre_row:[1,N]
        x_row_base = x_ptr + m * stride_xm
        h_row_base = hpre_ptr + m * N

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        x_blk_ptr = tl.make_block_ptr(
            base=x_row_base,
            shape=(N, D),
            strides=(stride_xn, stride_xd),
            offsets=(0, d_start),
            block_shape=(N, BLOCK_D),
            order=(1, 0),
        )
        h_blk_ptr = tl.make_block_ptr(
            base=h_row_base,
            shape=(1, N),
            strides=(N, 1),
            offsets=(0, 0),
            block_shape=(1, N),
            order=(1, 0),
        )

        # for n0 in range(0, N, BLOCK_N):
        x_blk = tl.load(x_blk_ptr, boundary_check=(0, 1)).to(tl.float32)  # [BN,BD]
        h_blk = tl.load(h_blk_ptr, boundary_check=(0, 1)).to(tl.float32)  # [1,BN]
        h_vec = tl.reshape(h_blk, (N,))                              # [BN]

        acc = tl.sum(x_blk * h_vec[:, None], axis=0)

        out_ptrs = out_ptr + m * D + d_offs
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=d_mask)


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
    x_flat = x.view(M, K).to(torch.float32)

    # biases/scales
    # alpha: 标量（这里用 float 作为 tl.constexpr；若你要运行时可变，我可以改成 runtime 参数）
    a0 = float(alpha_scales[0].item() if torch.is_tensor(alpha_scales[0]) else alpha_scales[0])
    a1 = float(alpha_scales[1].item() if torch.is_tensor(alpha_scales[1]) else alpha_scales[1])
    a2 = float(alpha_scales[2].item() if torch.is_tensor(alpha_scales[2]) else alpha_scales[2])

    bias0 = bias[0].contiguous().to(torch.float32)
    bias1 = bias[1].contiguous().to(torch.float32)
    bias2 = bias[2].contiguous().view(-1).to(torch.float32)

    # alloc outputs
    rsqrt = torch.empty((M,), device="npu:0", dtype=torch.float32)
    hpre_tmp = torch.empty((M, N), device="npu:0", dtype=torch.float32)
    hpost = torch.empty((M, N), device="npu:0", dtype=torch.float32)
    hcomb_flat = torch.empty((M, N * N), device="npu:0", dtype=torch.float32)
    h_in = torch.empty((M, D), device="npu:0", dtype=torch.bfloat16)

    num_cores = get_npu_properties()["num_aicore"]

    # kernel1: rsqrt
    rms_rsqrt_kernel[(num_cores,)](
        x_flat, rsqrt,
        M=M, K=K, num_cores=num_cores,
        BLOCK_M=128, BLOCK_K=128,
        norm_eps=norm_eps,
    )
    
    # kernel2: matmul + postprocess
    hc_matmul_post_kernel[(num_cores,)](
        x_flat, hc_weight,
        rsqrt,
        hpre_tmp, hpost, hcomb_flat,
        bias0, bias1, bias2,
        M=M, NT=NT, K=K, N=N,
        num_cores=num_cores,
        alpha0=a0, alpha1=a1, alpha2=a2,
        hc_eps=hc_eps,
        BLOCK_M=256, BLOCK_N=128, BLOCK_K=128,
    )

    # kernel3: h_in
    x_3d = x.view(M, N, D)
    h_in_kernel[(num_cores,)](
        x_3d, hpre_tmp, h_in,
        M=M, N=N, D=D,
        stride_xm=x_3d.stride(0),
        stride_xn=x_3d.stride(1),
        stride_xd=x_3d.stride(2),
        num_cores=num_cores,
        BLOCK_D=2048,
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
    torch.manual_seed(0)
    
    B, S, N, D = 2, 4096, 4, 1536 # D: 1536, 2048, 6144
    
    x = torch.randn((B, S, N, D), device=device, dtype=torch.bfloat16)
    hc_weight = torch.randn((N * D, N * (N + 2)), device=device, dtype=torch.float32)

    alpha_scales = [
        torch.tensor(1.0, device=device, dtype=torch.float32),
        torch.tensor(1.0, device=device, dtype=torch.float32),
        torch.tensor(1.0, device=device, dtype=torch.float32),
    ]
    bias = [
        torch.randn((N,), device=device, dtype=torch.float32),
        torch.randn((N,), device=device, dtype=torch.float32),
        torch.randn((N, N), device=device, dtype=torch.float32),
    ]
    return x, hc_weight, alpha_scales, bias


def check_accuracy():
    B, S, N, D = 2, 4096, 4, 1536 # D: 1536, 2048, 6144
    x, hc_weight, alpha_scales, bias = generate_test_data(B, S, N, D, device)
    
    h_in_t, hpost_t, hcomb_t = triton_hc_pre(x, hc_weight, alpha_scales, bias)
    h_in_g, hpost_g, hcomb_g = hc_pre_reference(x, hc_weight, alpha_scales, bias)
    
    # 对齐 dtype（h_in 是 bf16；其它 fp32）
    assert h_in_t.dtype == torch.bfloat16
    assert hpost_t.dtype == torch.float32
    assert hcomb_t.dtype == torch.float32
    
    assert_close_like_example(h_in_t.float(), h_in_g.float(), "h_in(bf16->fp32)", atol=2**-5, rtol=2**-5)
    assert_close_like_example(hpost_t, hpost_g, "H_post(fp32)", atol=2**-6, rtol=2**-6)
    assert_close_like_example(hcomb_t, hcomb_g, "H_comb_before(fp32)", atol=2**-6, rtol=2**-6)
    
    print("All checks passed.")
    
    return

def run_profiler():
    B, S, N, D = 2, 4096, 4, 1536 # D: 1536, 2048, 6144
    x, hc_weight, alpha_scales, bias = generate_test_data(B, S, N, D, device)
    
    print("Profiling Triton mHC-Pre...")
    profiler_wrapper(triton_hc_pre, x, hc_weight, alpha_scales, bias)
    
    # print("Profiling Golden mHC-Pre...")
    # profiler_wrapper(hc_pre_reference, x, hc_weight, alpha_scales, bias)
    
    return

def run_benchmark():
    # cast: 90 us
    # hc_rms_rsqrt_kernel: 206 us
    # hc_matmul_post_kernel: 190 us
    # h_in_kernel: 292 us
    # total: 778 us (1.7 ms golden, 1.1 ms triton)
    
    B, S, N, D = 2, 4096, 4, 1536 # D: 1536, 2048, 6144
    x, hc_weight, alpha_scales, bias = generate_test_data(B, S, N, D, device)
    
    # triton
    h_in_t, hpost_t, hcomb_t, t_triton, m_triton = bench(triton_hc_pre, x, hc_weight, alpha_scales, bias)
    # golden
    h_in_g, hpost_g, hcomb_g, t_golden, m_golden = bench(hc_pre_reference, x, hc_weight, alpha_scales, bias)
    
    e_triton_O = rel_err(h_in_t.float(), h_in_g.float())
    
    # Display Results in Table
    console = Console()
    table = Table(title="mHC-Pre", box=box.SIMPLE_HEAVY)
    table.add_column("Method", justify="left", style="cyan", no_wrap=True)
    table.add_column("Time (ms)", justify="right", style="magenta")
    table.add_column("Memory (MB)", justify="right", style="green")
    table.add_column("RelErr O", justify="right", style="red")

    table.add_row("Golden mHC-Pre", f"{t_golden:.3f}", f"{m_golden:.1f}", "-")
    table.add_row("Triton mHC-Pre", f"{t_triton:.3f}", f"{m_triton:.1f}", f"{e_triton_O:.2e}")
    console.print(table)
    
    return


if __name__ == "__main__":
    torch.npu.set_device(0)
    device = torch.device("npu:0")

    check_accuracy()
    # run_profiler()
    # run_benchmark()
    