import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm_functional(x, gamma=1, norm_eps=1e-6):
    """
    Stateless RMSNorm for functional usage.
    x: [..., dim], FP32 expected
    gamma: [dim], Learnable scale
    """
    x = x.float()

    # 1. Square & Mean
    var = x.pow(2).mean(dim=-1, keepdim=True)
    # 2. Rsqrt
    scale = torch.rsqrt(var + norm_eps)
    # 3. Normalize & Scale
    x_norm = x * scale

    return (x_norm * gamma).to(torch.float32)


########################################
## Our Implementation
########################################
class mHC(nn.Module):
    def __init__(self):
        super().__init__()

    def hc_pre(self, x, hc_weight, alpha_scales, bias, norm_eps=1e-6, hc_eps=1e-6):
        """
        Operator 1: Hyper-Connection Generation
        Input:
            x [B, S, N, D] (BF16)
            hc_weight, alpha_scales, bias, hc_norm_gamma (External Params)
        Output:
            h_in (BF16), H_post (FP32), H_comb_before (FP32)
        """
        B, S, N, D = x.shape

        # 1. RMSNorm (Functional fix)
        # Flatten: [B, S, N, D] -> [B, S, N*D]
        X_flat = x.view(B, S, -1).float()
        rsqrt = torch.rsqrt(X_flat.square().mean(-1, keepdim=True) + norm_eps)
        # Cast to FP32 is handled inside helper, passing explicit gamma
        # X_norm = rms_norm_functional(X_flat, norm_eps=norm_eps)  # FP32

        # 2. MatMul
        # [B, S, ND] @ [ND, Total] -> [B, S, Total]
        X_hat = torch.matmul(X_flat, hc_weight)  # FP32
        X_hat = X_hat * rsqrt  # Scale

        # 3. Split
        # Splits: [N, N, N*N]
        splits = [N, N, N * N]
        X_pre, X_post, X_comb = torch.split(X_hat, splits, dim=-1)

        # --- Branch Pre ---
        # H_pre = sigmoid(alpha_scales * X + bias)
        H_pre = torch.sigmoid(alpha_scales[0] * X_pre + bias[0]) + hc_eps  # [B, S, N]

        # h_in logic: sum_n(H_pre * x)
        x_fp32 = x.float()
        weighted_X = H_pre.unsqueeze(-1) * x_fp32  # [B, S, N, D]
        h_in = weighted_X.sum(dim=2)  # Sum over N -> [B, S, D]
        h_in = h_in.to(torch.bfloat16)  # Cast back to BF16

        # --- Branch Post ---
        # H_post = 2 * sigmoid(alpha_scales * X + bias)
        H_post = 2.0 * torch.sigmoid(
            alpha_scales[1] * X_post + bias[1]
        )  # [B, S, N], FP32

        # --- Branch Comb (formerly Res) ---
        # Reshape [B, S, N*N] -> [B, S, N, N]
        X_comb = X_comb.view(B, S, N, N)
        # H_comb_before = alpha_scales * X + bias
        H_comb_before = alpha_scales[2] * X_comb + bias[2]  # [B, S, N, N]

        return h_in, H_post, H_comb_before

    def sinkhorn(self, H_comb_before, hc_eps=1e-6, hc_sinkhorn_iters=20):
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
            # Row Norm
            H_comb = H_comb / (H_comb.sum(dim=-1, keepdim=True) + hc_eps)
            # Col Norm
            H_comb = H_comb / (H_comb.sum(dim=-2, keepdim=True) + hc_eps)
            # print(f"softmax: iter={i}, row: (max={torch.max(H_comb.sum(dim=-1)):.4f}, min={torch.min(H_comb.sum(dim=-1)):.4f}); col(max={torch.max(H_comb.sum(dim=-2)):.4f}, min={torch.min(H_comb.sum(dim=-2)):.4f})")

        return H_comb

    def sinkhorn_knopps_log_version(self, h_res, sinkhorn_iters, eps=1e-6):
        # logsumexp稳定版
        # 行归一
        # h_res = F.log_softmax(h_res, dim=-1)
        # h_res = (torch.softmax(H_comb_before, dim=-1) + eps).log()
        h_res = torch.logaddexp(F.log_softmax(h_res, dim=-1), torch.tensor(eps).log())

        # 列归一
        h_res = h_res - torch.logsumexp(h_res, dim=-2, keepdim=True)
        # 交替行列
        for i in range(sinkhorn_iters - 1):
            h_res = h_res - torch.logsumexp(h_res, dim=-1, keepdim=True)
            h_res = h_res - torch.logsumexp(h_res, dim=-2, keepdim=True)
            # print(f"logsumexp: iter={i}, row: (max={torch.max(h_res.exp().sum(dim=-1)):.4f}, min={torch.min(h_res.exp().sum(dim=-1)):.4f}); col(max={torch.max(h_res.exp().sum(dim=-2)):.4f}, min={torch.min(h_res.exp().sum(dim=-2)):.4f})")

        return h_res.exp()

    def hc_post(self, h_out, H_post, H_comb, x):
        """
        Operator 2: Manifold Projection
        Input: h_out, H_post, H_comb, x
        Output: y (BF16)
        """
        h_out_fp32 = h_out.float()
        x_fp32 = x.float()

        # 1. Broadcast Mul
        h_post_term = H_post.unsqueeze(-1) * h_out_fp32.unsqueeze(2)

        # 2. BMM
        # H_comb [B, S, N, N].T @ x [B, S, N, D]
        # h_comb_term = torch.matmul(torch.transpose(H_comb, -1, -2), x_fp32)
        h_comb_term = torch.sum(H_comb.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=2)

        # 3. Add
        y = h_post_term + h_comb_term

        return y.to(torch.bfloat16)

    def forward(
        self,
        x,
        hc_weight,
        alpha_scales,
        bias,
        norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    ):
        """
        Full mHC Forward Pass
        Input:
            x [B, S, N, D] (BF16)
            hc_weight, alpha_scales, bias, hc_norm_gamma (External Params)
        Output:
            y (BF16)
        """
        # Op 1: Hyper-Connection Generation
        self.h_in, self.H_post, self.H_comb_before = self.hc_pre(
            x, hc_weight, alpha_scales, bias, norm_eps=norm_eps
        )

        # Sinkhorn Normalization
        self.H_comb = self.sinkhorn(self.H_comb_before, hc_eps, hc_sinkhorn_iters)

        # Here you would typically have some sub-layer processing on h_in
        # For demonstration, we'll just pass it through unchanged
        self.h_out = 2 * self.h_in  # Placeholder for sub-layer output

        # Op 2: Manifold Projection
        y = self.hc_post(self.h_out, self.H_post, self.H_comb, x)
        return y

########################################
## Deepseek
#########################################
def sinkhorn_knopps(h_res, sinkhorn_iters, eps):
    h_res = h_res.softmax(-1) + eps
    col_sum = h_res.sum(-2, keepdim=True)
    h_res = h_res / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = h_res.sum(-1, keepdim=True)
        h_res = h_res / (row_sum + eps)
        col_sum = h_res.sum(-2, keepdim=True)
        h_res = h_res / (col_sum + eps)
    return h_res


def sinkhorn_knopps_log_version(h_res, sinkhorn_iters, eps):
    # logsumexp稳定版
    # 行归一
    h_res = F.log_softmax(h_res, dim=-1)
    # 列归一
    h_res = h_res - torch.logsumexp(h_res, dim=-2, keepdim=True)
    # 交替行列
    for _ in range(sinkhorn_iters - 1):
        h_res = h_res - torch.logsumexp(h_res, dim=-1, keepdim=True)
        h_res = h_res - torch.logsumexp(h_res, dim=-2, keepdim=True)
    return h_res


def hc_split_sinkhorn_torch(
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    num_stream: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    h_pre, h_post, h_res = weight.split(
        [num_stream, num_stream, num_stream * num_stream], dim=-1
    )
    h_res = h_res.unflatten(-1, (num_stream, num_stream))

    h_pre = (
        F.sigmoid(h_pre * hc_scale[0] + hc_base[:num_stream].unsqueeze(0).unsqueeze(0))
        + eps
    )
    h_post = 2 * F.sigmoid(
        h_post * hc_scale[1]
        + hc_base[num_stream : 2 * num_stream].unsqueeze(0).unsqueeze(0)
    )
    h_res = h_res * hc_scale[2] + hc_base[2 * num_stream :].view(
        num_stream, num_stream
    ).unsqueeze(0).unsqueeze(0)

    h_res = sinkhorn_knopps(h_res, sinkhorn_iters, eps)
    return h_pre, h_post, h_res


class MHCModuler(nn.Module):
    def __init__(self, hidden_size, mhc_num_stream, sinkhorn_iters, rms_norm_eps):
        super().__init__()

        self.hc_eps = 1e-6  # magic number
        self.hidden_size = hidden_size
        self.num_stream = mhc_num_stream
        self.hc_sinkhorn_iters = sinkhorn_iters
        mix_hc = (2 + self.num_stream) * self.num_stream
        hc_dim = self.num_stream * self.hidden_size
        self.hc_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))

        self.norm_eps = rms_norm_eps

    def hc_pre(self, x: torch.Tensor):
        # x: [b, s, hc, d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b, s, hc, d]
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)

        # weight = F.linear(x * rsqrt, self.hc_fn)
        weight = F.linear(x, self.hc_fn) * rsqrt

        print(
            "dtype1:",
            weight.dtype,
            self.hc_fn.dtype,
            self.hc_scale.dtype,
            self.hc_base.dtype,
        )
        h_pre, h_post, h_res = hc_split_sinkhorn_torch(
            weight,
            self.hc_scale,
            self.hc_base,
            self.num_stream,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        print("dtype2:", h_pre.dtype, h_post.dtype, h_res.dtype)
        y = torch.sum(h_pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), h_post, h_res

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ):
        # x: [b, s, d], residual: [b, s, hc, d], h_post: [b, s, hc], h_res: [b, s, hc, hc], y:[b, s, hc, d]
        # print data types
        y = h_post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            h_res.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)


def rmse(O, O_golden):
    relative_l2_error = torch.linalg.norm(O - O_golden) / (
        torch.linalg.norm(O_golden) + torch.finfo(torch.float32).eps
    )
    return relative_l2_error


# --- Test Script ---
if __name__ == "__main__":
    torch.manual_seed(42)

    # Configuration
    B, S, N, D = 2, int(4 * 1024), 4, int(128)
    device = "npu" if torch.cuda.is_available() else "cpu"
    
    # Create Dummy Input (BF16)
    x = torch.randn(B, S, N, D, device=device, dtype=torch.bfloat16)
    print(f"Input x shape: {x.shape}, dtype: {x.dtype}")

    # --- Run Operator 1 ---
    # Initialize alpha_scales to 1.0, S to 0.01
    alpha_scales = torch.tensor([1.0, 1.0, 1.0], device=device)
    bias = [
        torch.full((N,), 0.01, device=device),  # S_1
        torch.full((N,), 0.01, device=device),  # S_2
        torch.full((N, N), 0.01, device=device),  # S_3
    ]

    # 初始化做了 1/sqrt(ND) 级别的缩放:防止 exp() 大面积下溢成 0，Sinkhorn 出现“零行/零列”
    hc_weight = torch.randn(N * D, N + N + N * N, device=device)  # / (N * D) ** 0.5

    mhc = mHC().to(device)
    y = mhc.forward(
        x,
        hc_weight,
        alpha_scales,
        bias,
        norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )

    x0 = x
    mhc2 = MHCModuler(
        hidden_size=D, mhc_num_stream=N, sinkhorn_iters=20, rms_norm_eps=1e-6
    )
    # set mhc base and scale to match above
    mhc2.hc_base.data = torch.tensor([0.01] * (2 * N + N * N))
    mhc2.hc_scale.data = torch.tensor([1.0, 1.0, 1.0])
    mhc2.hc_fn.data = hc_weight.T
    x, h_post, h_res = mhc2.hc_pre(x)
    new_x = mhc2.hc_post(2 * x, x0, h_post, h_res)

    # check if x equals h_in
    print("\n--- Checking hc_pre outputs ---")
    print("Relative L2 Error (RMSE) h_in:", rmse(x.float(), mhc.h_in.float()).item())
    print(
        "Relative L2 Error (RMSE) H_post:",
        rmse(h_post.float(), mhc.H_post.float()).item(),
    )
    print(
        "Relative L2 Error (RMSE) H_res:",
        rmse(h_res.float(), mhc.H_comb.float()).item(),
    )
    print("Relative L2 Error (RMSE):", rmse(new_x.float(), y.float()).item())