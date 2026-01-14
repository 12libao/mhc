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


def bench(fn, *args):
    iters=10
    warmup=5
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

    # 如果out是元组，展开它；否则直接返回
    if isinstance(out, tuple):
        return (*out, elapsed_time_us, active_mem_mb)
    else:
        return (out, elapsed_time_us, active_mem_mb)

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


def sinkhorn_reference(H_comb_before, hc_eps=1e-6, hc_sinkhorn_iters=20):
    """
    Sinkhorn Normalization (A-scheme)
    Input : H_comb_before [B, S, N, N]
    Output: H_comb        [B, S, N, N] (approximately doubly stochastic)
    """
    # 1) Row-wise softmax, then add eps to every entry
    H_comb = torch.softmax(H_comb_before, dim=-1) + hc_eps

    # 2) Initial column normalization
    H_comb = H_comb / (H_comb.sum(dim=-2, keepdim=True) + hc_eps)

    # 3) Alternate normalization: (row -> col), repeated (iters - 1) times
    for i in range(max(hc_sinkhorn_iters - 1, 0)):
        H_comb = H_comb / (H_comb.sum(dim=-1, keepdim=True) + hc_eps) # Row Norm
        H_comb = H_comb / (H_comb.sum(dim=-2, keepdim=True) + hc_eps) # Col Norm
        
    return H_comb

def hc_post_reference(x, h_out, H_post, H_comb):
    """
    Operator 2: Manifold Projection
    Input: 
        h_out: [B, S, D] (BF16) 
        H_post: [B, S, N] (FP32)
        H_comb: [B, S, N, N] (FP32)
        x: [B, S, N, D] (BF16)
    Output: y (BF16)
    """
    h_out_fp32 = h_out.float()
    x_fp32 = x.float()

    # 1. Broadcast Mul
    h_post_term = H_post.unsqueeze(-1) * h_out_fp32.unsqueeze(2) # [B, S, N, D]

    # 2. BMM
    # H_comb [B, S, N, N].T @ x [B, S, N, D]
    # h_comb_term = torch.matmul(torch.transpose(H_comb, -1, -2), x_fp32)
    h_comb_term = torch.sum(H_comb.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=2)

    # 3. Add
    y = h_post_term + h_comb_term

    return y.to(torch.bfloat16)
    
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
def hc_pre_triton(x, hc_weight, alpha_scales, bias, norm_eps=1e-6, hc_eps=1e-6):
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


@triton.jit
def hc_post_kernel(
    x_ptr,        # [M, 4, D] bf16
    h_out_ptr,    # [M, D] bf16
    H_post_ptr,   # [M, 4] fp32
    H_comb_ptr,   # [M, 4, 4] fp32  (index: [k, n])
    y_ptr,        # [M, 4, D] bf16
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)  # 0..M-1
    pid_d = tl.program_id(1)  # 0..ceil_div(D, BLOCK_D)-1

    m = pid_m
    d0 = pid_d * BLOCK_D
    d = d0 + tl.arange(0, BLOCK_D)
    d_mask = d < D

    # n = 0..3
    n = tl.arange(0, 4)

    # base offsets
    base_hout  = m * D
    base_hpost = m * 4
    base_hcomb = m * 16          # 4*4
    base_x     = m * (4 * D)
    base_y     = m * (4 * D)

    # load h_out[d]
    hout = tl.load(h_out_ptr + base_hout + d, mask=d_mask, other=0).to(tl.float32)  # [BD]

    # load H_post[n]
    hpost = tl.load(H_post_ptr + base_hpost + n).to(tl.float32)  # [4]

    # acc[n, d] = H_post[n] * h_out[d]
    acc = hpost[:, None] * hout[None, :]  # [4, BD]

    # add sum_k H_comb[k, n] * x[k, d], k=0..3 (完全展开，不跨 program 归约)
    # k = 0
    x0 = tl.load(x_ptr + base_x + 0 * D + d, mask=d_mask, other=0).to(tl.float32)  # [BD]
    w0 = tl.load(H_comb_ptr + base_hcomb + 0 * 4 + n).to(tl.float32)               # [4]
    acc += w0[:, None] * x0[None, :]

    # k = 1
    x1 = tl.load(x_ptr + base_x + 1 * D + d, mask=d_mask, other=0).to(tl.float32)
    w1 = tl.load(H_comb_ptr + base_hcomb + 1 * 4 + n).to(tl.float32)
    acc += w1[:, None] * x1[None, :]

    # k = 2
    x2 = tl.load(x_ptr + base_x + 2 * D + d, mask=d_mask, other=0).to(tl.float32)
    w2 = tl.load(H_comb_ptr + base_hcomb + 2 * 4 + n).to(tl.float32)
    acc += w2[:, None] * x2[None, :]

    # k = 3
    x3 = tl.load(x_ptr + base_x + 3 * D + d, mask=d_mask, other=0).to(tl.float32)
    w3 = tl.load(H_comb_ptr + base_hcomb + 3 * 4 + n).to(tl.float32)
    acc += w3[:, None] * x3[None, :]

    # store y[n, d]
    y_ptrs = y_ptr + base_y + n[:, None] * D + d[None, :]
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=d_mask[None, :])
    
    
def hc_post_triton(x, h_out, H_post, H_comb):
    """
    mHC Post Triton Implementation
    Input: 
        h_out: [B, S, D] (BF16) 
        H_post: [B, S, N] (FP32)
        H_comb: [B, S, N, N] (FP32)
        x: [B, S, N, D] (BF16)
    Output: y (BF16)
    """
    assert x.is_npu and h_out.is_npu and H_post.is_npu and H_comb.is_npu
    x = x.contiguous()
    h_out = h_out.contiguous()
    H_post = H_post.contiguous()
    H_comb = H_comb.contiguous()

    B, S, N, D = x.shape
    M = B * S
    K = N * D

    # flatten/view
    x_flat_MND = x.view(M, N, D)
    h_out_flat_MD = h_out.view(M, D)
    H_post_flat_MN = H_post.view(M, N)
    H_comb_flat_MNN = H_comb.view(M, N, N)

    # alloc output
    y_flat = torch.empty((M, 4, D), device=h_out.device, dtype=torch.bfloat16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(D, META['BLOCK_D']))
    
    # launch kernel
    grid = lambda META: (M, triton.cdiv(D, META["BLOCK_D"]))
    grid = lambda META: (M,)
        
    hc_post_kernel1[grid](
        x_flat_MND, 
        h_out_flat_MD, 
        H_post_flat_MN, 
        H_comb_flat_MNN, 
        y_flat,
        M=M, D=D,
        BLOCK_D=min(3072, D),
    )

    return y_flat.view(B, S, 4, D)


@triton.jit
def hc_post_kernel1(
    x_ptr,        # [M, 4, D] bf16
    h_out_ptr,    # [M, D] bf16
    H_post_ptr,   # [M, 4] fp32
    H_comb_ptr,   # [M, 4, 4] fp32  (index: [k, n])
    y_ptr,        # [M, 4, D] bf16
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)   # one program handles one m = (b,s)
    m = pid
    if m >= M:
        return

    # n=0..3
    n = tl.arange(0, 4)

    base_hout  = m * D
    base_hpost = m * 4
    base_hcomb = m * 16          # 4*4
    base_x     = m * (4 * D)
    base_y     = m * (4 * D)

    # H_post[n] (fp32)
    hpost = tl.load(H_post_ptr + base_hpost + n).to(tl.float32)  # [4]

    # H_comb[k,n] 全部预加载（fp32），后面每个 D-tile 复用
    w0 = tl.load(H_comb_ptr + base_hcomb + 0 * 4 + n).to(tl.float32)  # [4]
    w1 = tl.load(H_comb_ptr + base_hcomb + 1 * 4 + n).to(tl.float32)  # [4]
    w2 = tl.load(H_comb_ptr + base_hcomb + 2 * 4 + n).to(tl.float32)  # [4]
    w3 = tl.load(H_comb_ptr + base_hcomb + 3 * 4 + n).to(tl.float32)  # [4]

    # kernel 内部切 D
    for d0 in tl.static_range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        d_mask = d < D

        # h_out[d]
        hout = tl.load(h_out_ptr + base_hout + d, mask=d_mask, other=0).to(tl.float32)  # [BD]

        # h_post_term: H_post[n] * h_out[d]
        acc = hpost[:, None] * hout[None, :]  # [4, BD]

        # x[k, d] + comb term（k=0..3 完全展开）
        x0 = tl.load(x_ptr + base_x + 0 * D + d, mask=d_mask, other=0).to(tl.float32)
        x1 = tl.load(x_ptr + base_x + 1 * D + d, mask=d_mask, other=0).to(tl.float32)
        x2 = tl.load(x_ptr + base_x + 2 * D + d, mask=d_mask, other=0).to(tl.float32)
        x3 = tl.load(x_ptr + base_x + 3 * D + d, mask=d_mask, other=0).to(tl.float32)

        acc += w0[:, None] * x0[None, :]
        acc += w1[:, None] * x1[None, :]
        acc += w2[:, None] * x2[None, :]
        acc += w3[:, None] * x3[None, :]

        # store y[n, d]
        y_ptrs = y_ptr + base_y + n[:, None] * D + d[None, :]
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=d_mask[None, :])

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


def check_accuracy_pre(x, hc_weight, alpha_scales, bias):
    
    stream = torch.npu.current_stream()
    stream.synchronize()
    h_in_t, hpost_t, hcomb_t = hc_pre_triton(x, hc_weight, alpha_scales, bias)
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

def check_accuracy_post(x, hc_weight, alpha_scales, bias):
    
    stream = torch.npu.current_stream()
    stream.synchronize()
    h_in, h_post, h_comb_before = hc_pre_reference(x, hc_weight, alpha_scales, bias)
    H_comb = sinkhorn_reference(h_comb_before)
    
    y_g = hc_post_reference(x, h_in, h_post, H_comb)
    y_t = hc_post_triton(x, h_in, h_post, H_comb)
    
    stream.synchronize()
    
    # 对齐 dtype
    assert h_in.dtype == torch.bfloat16
    assert h_post.dtype == torch.float32
    assert H_comb.dtype == torch.float32
    assert y_t.dtype == torch.bfloat16
    assert y_g.dtype == torch.bfloat16
    
    assert_close_like_example(y_t.float(), y_g.float(), "h_in(bf16->fp32)", atol=2**-5, rtol=2**-5)
    
    print("All checks passed.")
    
    return

def run_profiler_pre(x, hc_weight, alpha_scales, bias):
    
    profiler_wrapper(hc_pre_triton, x, hc_weight, alpha_scales, bias)
    
    # profiler_wrapper(hc_pre_reference, x, hc_weight, alpha_scales, bias)
    
    return

def run_profiler_post(x, hc_weight, alpha_scales, bias):
    
    h_in, h_post, h_comb_before = hc_pre_reference(x, hc_weight, alpha_scales, bias)
    H_comb = sinkhorn_reference(h_comb_before)
    
    # profiler_wrapper(hc_post_triton, x, h_in, h_post, H_comb)
    
    profiler_wrapper(hc_post_reference, x, h_in, h_post, H_comb)
    
    return

def run_benchmark_pre(x, hc_weight, alpha_scales, bias):
   
    h_in_t, hpost_t, hcomb_t, t_triton, m_triton = bench(hc_pre_triton, x, hc_weight, alpha_scales, bias)
    h_in_g, hpost_g, hcomb_g, t_golden, m_golden = bench(hc_pre_reference, x, hc_weight, alpha_scales, bias)
    
    e_triton = rel_err(h_in_t.float(), h_in_g.float())
    
    # Display Results in Table
    console = Console()
    table = Table(title="\nmHC-Pre: B:{}, S:{}, N:{}, D:{}".format(*x.shape), box=box.SIMPLE_HEAVY)
    table.add_column("Method", justify="left", style="cyan", no_wrap=True)
    table.add_column("Time (us)", justify="right", style="magenta")
    table.add_column("Memory (MB)", justify="right", style="green")
    table.add_column("RelErr", justify="right", style="red")

    table.add_row("Golden mHC-Pre", f"{t_golden:.0f}", f"{m_golden:.1f}", "-")
    table.add_row("Triton mHC-Pre", f"{t_triton:.0f}", f"{m_triton:.1f}", f"{e_triton:.2e}")
    console.print(table)
    
    return

def run_benchmark(x, hc_weight, alpha_scales, bias):
   
    h_in, h_post, h_comb_before = hc_pre_reference(x, hc_weight, alpha_scales, bias)
    H_comb = sinkhorn_reference(h_comb_before)
    
    y_t, t_triton, m_triton = bench(hc_post_triton, x, h_in, h_post, H_comb)
    y_g, t_golden, m_golden = bench(hc_post_reference, x, h_in, h_post, H_comb)
    
    e_triton = rel_err(y_t.float(), y_g.float())
    
    # Display Results in Table
    console = Console()
    table = Table(title="\nmHC-Post: B:{}, S:{}, N:{}, D:{}".format(*x.shape), box=box.SIMPLE_HEAVY)
    table.add_column("Method", justify="left", style="cyan", no_wrap=True)
    table.add_column("Time (us)", justify="right", style="magenta")
    table.add_column("Memory (MB)", justify="right", style="green")
    table.add_column("RelErr", justify="right", style="red")

    table.add_row("Golden mHC-Post", f"{t_golden:.0f}", f"{m_golden:.1f}", "-")
    table.add_row("Triton mHC-Post", f"{t_triton:.0f}", f"{m_triton:.1f}", f"{e_triton:.2e}")
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

    # check_accuracy_pre(x, hc_weight, alpha_scales, bias)
    # run_profiler_pre(x, hc_weight, alpha_scales, bias)
    # run_benchmark_pre(x, hc_weight, alpha_scales, bias)
    
    # check_accuracy_post(x, hc_weight, alpha_scales, bias)
    # run_profiler_post(x, hc_weight, alpha_scales, bias)
    run_benchmark(x, hc_weight, alpha_scales, bias)
    