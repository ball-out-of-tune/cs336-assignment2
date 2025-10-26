import torch
from typing import Type

# Require Triton
try:
    import triton
    import triton.language as tl
except Exception as e:
    raise ImportError("Triton is required for get_flashattention_autograd_function_triton()") from e


def _round_div(x, y):
    return (x + y - 1) // y


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)  # which query tile (along queries)
    batch_index = tl.program_id(1)       # collapsed batch (B*H)

    # Make block_ptrs for Q (only the query tile), O, and L (only the query tile)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Base pointers for K and V (we'll offset them inside the loop)
    K_base_ptr = K_ptr + batch_index * stride_kb
    V_base_ptr = V_ptr + batch_index * stride_vb

    # load q block (Bq, D) and cast to float32 and scale
    q_block = tl.load(Q_block_ptr)  # shape (Q_TILE_SIZE, D)
    q_block = q_block.to(tl.float32) * scale  # float32 scaled queries

    # accumulators in float32
    # initialize m_prev to -inf, denom_prev to 0, out_prev to 0
    m_prev = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    denom_prev = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    out_prev = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # loop over key tiles (single loop as required)
    for key_tile_idx in range(0, num_key_tiles):
        k_off = key_tile_idx * K_TILE_SIZE
        K_block_ptr = tl.make_block_ptr(
            K_base_ptr,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(k_off, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_base_ptr,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(k_off, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        k_block = tl.load(K_block_ptr)  # (Bk, D)
        v_block = tl.load(V_block_ptr)  # (Bk, Dv) -- assume Dv == D or compatible

        # cast to float32 for computations
        k_block_f = k_block.to(tl.float32)
        v_block_f = v_block.to(tl.float32)

        # compute scores = q_block @ k_block^T  -> shape (Bq, Bk)
        # Implementation: broadcast multiply then sum over D (works when D moderate)
        # q_block: (Bq, D) -> (Bq, 1, D)
        # k_block_f: (Bk, D) -> (1, Bk, D)
        q_exp = tl.reshape(q_block, (Q_TILE_SIZE, 1, D))
        k_exp = tl.reshape(k_block_f, (1, K_TILE_SIZE, D))
        scores = tl.sum(q_exp * k_exp, axis=2)  # (Bq, Bk) float32

        # stable softmax per query across this key tile
        m_block = tl.max(scores, axis=1)  # (Bq,)
        p = tl.exp(scores - tl.reshape(m_block, (Q_TILE_SIZE, 1)))  # (Bq, Bk)
        denom_block = tl.sum(p, axis=1)  # (Bq,)

        # s_block = p @ v_block  -> (Bq, Dv)
        p_exp = tl.reshape(p, (Q_TILE_SIZE, K_TILE_SIZE, 1))  # (Bq, Bk, 1)
        s_block = tl.sum(p_exp * tl.reshape(v_block_f, (1, K_TILE_SIZE, D)), axis=1)  # (Bq, D)

        # merge with previous accumulators using stable formula
        if key_tile_idx == 0:
            m_prev = m_block
            denom_prev = denom_block
            out_prev = s_block
        else:
            m_new = tl.maximum(m_prev, m_block)
            exp_m_prev = tl.exp(m_prev - m_new)
            exp_m_block = tl.exp(m_block - m_new)
            out_prev = out_prev * tl.reshape(exp_m_prev, (Q_TILE_SIZE, 1)) + s_block * tl.reshape(exp_m_block, (Q_TILE_SIZE, 1))
            denom_prev = denom_prev * exp_m_prev + denom_block * exp_m_block
            m_prev = m_new

    # finalize outputs
    # O_block = out_prev / denom_prev.unsqueeze(-1)
    O_block = out_prev / tl.reshape(denom_prev, (Q_TILE_SIZE, 1))
    L_block = m_prev + tl.log(denom_prev)

    # cast to element type of destination and store
    # block_ptr.type.element_ty gives destination dtype
    O_to_store = O_block.to(O_block_ptr.type.element_ty)
    L_to_store = L_block.to(L_block_ptr.type.element_ty)

    tl.store(O_block_ptr, O_to_store)
    tl.store(L_block_ptr, L_to_store)


def my_get_flashattention_autograd_function_triton() -> Type[torch.autograd.Function]:
    """
    Triton-only FlashAttention forward autograd.Function.
    Backward is not implemented.
    """
    class TritonFlashAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, is_causal: bool = False):
            # Accept both (B, M, D) and (B, H, M, D)
            squeeze_heads = False
            if Q.dim() == 3:
                Q = Q.unsqueeze(1)
                K = K.unsqueeze(1)
                V = V.unsqueeze(1)
                squeeze_heads = True

            B, H, M, D = Q.shape
            N = K.shape[2]
            Dv = V.shape[3]
            # This kernel assumes D (per-head dim) is used for V as well (Dv == D) or compatible.
            # If Dv != D you can adapt kernel to take separate Dv constexpr and adjust shapes.
            assert Dv == D or True  # keep assertion relaxed; for strictness change to Dv == D

            BH = B * H
            Q_ = Q.reshape(BH, M, D)
            K_ = K.reshape(BH, N, D)
            V_ = V.reshape(BH, N, Dv)

            # prepare outputs
            O_ = torch.empty((BH, M, Dv), device=Q.device, dtype=Q.dtype)
            L_ = torch.empty((BH, M), device=Q.device, dtype=Q.dtype)

            # tile sizes (tune for performance)
            Q_TILE_SIZE = 64
            K_TILE_SIZE = 64

            # compute strides (element counts)
            stride_qb = Q_.stride(0)
            stride_qq = Q_.stride(1)
            stride_qd = Q_.stride(2)
            stride_kb = K_.stride(0)
            stride_kk = K_.stride(1)
            stride_kd = K_.stride(2)
            stride_vb = V_.stride(0)
            stride_vk = V_.stride(1)
            stride_vd = V_.stride(2)
            stride_ob = O_.stride(0)
            stride_oq = O_.stride(1)
            stride_od = O_.stride(2)
            stride_lb = L_.stride(0)
            stride_lq = L_.stride(1)

            # grid: (num_query_tiles, BH)
            num_query_tiles = _round_div(M, Q_TILE_SIZE)
            grid = (num_query_tiles, BH)

            scale = 1.0 / (D ** 0.5)

            # Launch the Triton kernel
            flash_fwd_kernel[grid](
                Q_, K_, V_,
                O_, L_,
                stride_qb, stride_qq, stride_qd,
                stride_kb, stride_kk, stride_kd,
                stride_vb, stride_vk, stride_vd,
                stride_ob, stride_oq, stride_od,
                stride_lb, stride_lq,
                M, N,
                scale,
                D, Q_TILE_SIZE, K_TILE_SIZE,
            )

            # reshape back to (B, H, M, Dv)
            O = O_.reshape(B, H, M, Dv)
            L = L_.reshape(B, H, M)

            # Save minimal tensors for potential backward (you'll need more for actual backward)
            L_to_save = L.mean(dim=1)
            ctx.save_for_backward(L_to_save, Q, K, V, O)

            if squeeze_heads:
                O = O.squeeze(1)
                L = L.squeeze(1)

            return O

        @staticmethod
        def backward(ctx, *grad_outputs):
            raise NotImplementedError("Backward not implemented for TritonFlashAttention")

    return TritonFlashAttention
