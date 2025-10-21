import torch
from typing import Type

def my_get_flashattention_autograd_function_pytorch() -> Type[torch.autograd.Function]:
    class FlashAttentionAutogradFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, is_causal: bool = False):
            # Support both (B, M, D) and (B, H, M, D)
            if Q.dim() == 3:
                Q = Q.unsqueeze(1)
                K = K.unsqueeze(1)
                V = V.unsqueeze(1)
                squeeze_heads = True
            else:
                squeeze_heads = False

            B, H, M, D = Q.shape
            N = K.shape[2]
            Dv = V.shape[3]

            tile_m, tile_n = 16, 16
            scale = 1.0 / (D ** 0.5)

            BH = B * H
            Q_ = Q.reshape(BH, M, D)
            K_ = K.reshape(BH, N, D)
            V_ = V.reshape(BH, N, Dv)

            O_ = torch.zeros((BH, M, Dv), device=Q.device, dtype=Q.dtype)
            L_ = torch.empty((BH, M), device=Q.device, dtype=Q.dtype)

            for i in range(0, M, tile_m):
                q = Q_[:, i:i+tile_m, :]
                t_m = q.shape[1]

                m_prev = torch.full((BH, t_m), float("-inf"), device=Q.device, dtype=Q.dtype)
                denom_prev = torch.zeros((BH, t_m), device=Q.device, dtype=Q.dtype)
                out_prev = torch.zeros((BH, t_m, Dv), device=Q.device, dtype=Q.dtype)

                for j in range(0, N, tile_n):
                    k = K_[:, j:j+tile_n, :]
                    v = V_[:, j:j+tile_n, :]
                    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                    m_block, _ = scores.max(dim=2)
                    p = torch.exp(scores - m_block.unsqueeze(-1))
                    denom_block = p.sum(dim=2)
                    s_block = torch.matmul(p, v)

                    if j == 0:
                        m_prev = m_block
                        denom_prev = denom_block
                        out_prev = s_block
                    else:
                        m_new = torch.maximum(m_prev, m_block)
                        exp_m_prev = torch.exp(m_prev - m_new)
                        exp_m_block = torch.exp(m_block - m_new)
                        out_prev = out_prev * exp_m_prev.unsqueeze(-1) + s_block * exp_m_block.unsqueeze(-1)
                        denom_prev = denom_prev * exp_m_prev + denom_block * exp_m_block
                        m_prev = m_new

                O_block = out_prev / denom_prev.unsqueeze(-1)
                O_[:, i:i+t_m, :] = O_block
                L_[:, i:i+t_m] = m_prev + torch.log(denom_prev)

            O = O_.reshape(B, H, M, Dv)
            L = L_.reshape(B, H, M)

            # Save tensors — ensure L is (B, M)
            L_to_save = L.mean(dim=1)  # or L[:, 0, :]
            ctx.save_for_backward(L_to_save, Q, K, V, O)

            if squeeze_heads:
                O = O.squeeze(1)
                L = L.squeeze(1)

            return O

        @staticmethod
        def backward(ctx, *grad_outputs):
            raise NotImplementedError("Backward not implemented")
    return FlashAttentionAutogradFunction
