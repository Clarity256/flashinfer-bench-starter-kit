"""
DeepSeek Sparse Attention TopK indexer (dsa_topk_indexer_fp8_h64_d128_topk2048_ps64).

Semantics follow the definition reference:
1) Decode paged FP8 KV index cache (DeepGEMM layout).
2) Compute score[t] = sum_h relu(dot(q[h], k[t])) * weight[h].
3) Return top-2048 global token indices per batch item (pad with -1).
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 64
HEAD_DIM_WITH_SCALE = 132
TOPK = 2048


if _HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_TOKENS": 64, "BLOCK_HEADS": 8, "BLOCK_D": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_TOKENS": 128, "BLOCK_HEADS": 8, "BLOCK_D": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_TOKENS": 128, "BLOCK_HEADS": 16, "BLOCK_D": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_TOKENS": 256, "BLOCK_HEADS": 8, "BLOCK_D": 32}, num_warps=8, num_stages=3),
        ],
        key=["seq_len"],
    )
    @triton.jit
    def _weighted_relu_score_kernel(
        q_ptr,
        k_ptr,
        w_ptr,
        out_ptr,
        seq_len,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kd,
        stride_out,
        BLOCK_TOKENS: tl.constexpr,
        BLOCK_HEADS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        pid_tok = tl.program_id(0)
        tok_offsets = pid_tok * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        tok_mask = tok_offsets < seq_len
        acc_tok = tl.zeros((BLOCK_TOKENS,), dtype=tl.float32)

        for h_start in range(0, NUM_HEADS, BLOCK_HEADS):
            head_offsets = h_start + tl.arange(0, BLOCK_HEADS)
            head_mask = head_offsets < NUM_HEADS
            dot_hk = tl.zeros((BLOCK_HEADS, BLOCK_TOKENS), dtype=tl.float32)

            for d_start in range(0, HEAD_DIM, BLOCK_D):
                d_offsets = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offsets < HEAD_DIM

                q_ptrs = q_ptr + head_offsets[:, None] * stride_qh + d_offsets[None, :] * stride_qd
                q_tile = tl.load(
                    q_ptrs,
                    mask=head_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                k_ptrs = k_ptr + tok_offsets[None, :] * stride_kt + d_offsets[:, None] * stride_kd
                k_tile = tl.load(
                    k_ptrs,
                    mask=d_mask[:, None] & tok_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                dot_hk += tl.dot(q_tile, k_tile)

            dot_hk = tl.maximum(dot_hk, 0.0)
            w = tl.load(w_ptr + head_offsets, mask=head_mask, other=0.0).to(tl.float32)
            acc_tok += tl.sum(dot_hk * w[:, None], axis=0)

        tl.store(out_ptr + tok_offsets * stride_out, acc_tok, mask=tok_mask)


def _check_inputs(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> None:
    if q_index_fp8.ndim != 3:
        raise ValueError(f"q_index_fp8 must be rank-3, got {q_index_fp8.shape}")
    if k_index_cache_fp8.ndim != 4:
        raise ValueError(f"k_index_cache_fp8 must be rank-4, got {k_index_cache_fp8.shape}")
    if weights.ndim != 2:
        raise ValueError(f"weights must be rank-2, got {weights.shape}")
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must be rank-1, got {seq_lens.shape}")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be rank-2, got {block_table.shape}")

    batch_size, num_heads, head_dim = q_index_fp8.shape
    num_pages, page_size, _, head_dim_with_scale = k_index_cache_fp8.shape
    if num_heads != NUM_INDEX_HEADS:
        raise ValueError(f"num_index_heads must be {NUM_INDEX_HEADS}, got {num_heads}")
    if head_dim != INDEX_HEAD_DIM:
        raise ValueError(f"index_head_dim must be {INDEX_HEAD_DIM}, got {head_dim}")
    if page_size != PAGE_SIZE:
        raise ValueError(f"page_size must be {PAGE_SIZE}, got {page_size}")
    if head_dim_with_scale != HEAD_DIM_WITH_SCALE:
        raise ValueError(f"head_dim_with_scale must be {HEAD_DIM_WITH_SCALE}, got {head_dim_with_scale}")

    if weights.shape != (batch_size, NUM_INDEX_HEADS):
        raise ValueError(
            f"weights shape must be {(batch_size, NUM_INDEX_HEADS)}, got {weights.shape}"
        )
    if seq_lens.shape[0] != batch_size:
        raise ValueError(f"seq_lens length must be {batch_size}, got {seq_lens.shape[0]}")
    if block_table.shape[0] != batch_size:
        raise ValueError(f"block_table batch size must be {batch_size}, got {block_table.shape[0]}")
    if num_pages <= 0:
        raise ValueError("num_pages must be positive")


def _decode_selected_pages(
    kv_flat_u8: torch.Tensor,  # [num_pages, PAGE_SIZE * HEAD_DIM_WITH_SCALE], uint8
    page_indices: torch.Tensor,  # [num_pages_for_seq], int64
) -> torch.Tensor:
    selected = kv_flat_u8.index_select(0, page_indices)

    fp8_bytes = selected[:, : PAGE_SIZE * INDEX_HEAD_DIM].contiguous()
    fp8 = fp8_bytes.view(-1, PAGE_SIZE, INDEX_HEAD_DIM).view(torch.float8_e4m3fn)
    fp8_f32 = fp8.to(torch.float32)

    scale_bytes = selected[:, PAGE_SIZE * INDEX_HEAD_DIM :].contiguous()
    scales = scale_bytes.view(-1, PAGE_SIZE, 4).view(torch.float32)
    return fp8_f32 * scales


def _weighted_relu_scores_torch(q_b: torch.Tensor, k_tokens: torch.Tensor, w_b: torch.Tensor) -> torch.Tensor:
    scores = q_b @ k_tokens.transpose(0, 1)
    scores = torch.relu(scores)
    scores = scores * w_b[:, None]
    return scores.sum(dim=0)


def _weighted_relu_scores_triton(q_b: torch.Tensor, k_tokens: torch.Tensor, w_b: torch.Tensor) -> torch.Tensor:
    if (
        not _HAS_TRITON
        or not q_b.is_cuda
        or not k_tokens.is_cuda
        or not w_b.is_cuda
        or os.environ.get("DSA_INDEXER_DISABLE_TRITON") == "1"
    ):
        return _weighted_relu_scores_torch(q_b, k_tokens, w_b)

    q_b = q_b.contiguous()
    k_tokens = k_tokens.contiguous()
    w_b = w_b.contiguous()

    seq_len = k_tokens.shape[0]
    out = torch.empty((seq_len,), device=q_b.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_TOKENS"]),)
    _weighted_relu_score_kernel[grid](
        q_b,
        k_tokens,
        w_b,
        out,
        seq_len,
        q_b.stride(0),
        q_b.stride(1),
        k_tokens.stride(0),
        k_tokens.stride(1),
        out.stride(0),
        NUM_HEADS=NUM_INDEX_HEADS,
        HEAD_DIM=INDEX_HEAD_DIM,
    )
    return out


@torch.no_grad()
def run(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> Tuple[torch.Tensor]:
    _check_inputs(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)

    batch_size = q_index_fp8.shape[0]
    num_pages = k_index_cache_fp8.shape[0]
    device = q_index_fp8.device

    q_f32 = q_index_fp8.to(torch.float32)
    w_f32 = weights.to(torch.float32)

    kv_flat_u8 = k_index_cache_fp8.view(torch.uint8).reshape(
        num_pages, PAGE_SIZE * HEAD_DIM_WITH_SCALE
    )

    topk_indices = torch.full((batch_size, TOPK), -1, dtype=torch.int32, device=device)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len <= 0:
            continue

        num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
        k_paged = _decode_selected_pages(kv_flat_u8, page_indices)
        k_tokens = k_paged.reshape(-1, INDEX_HEAD_DIM)[:seq_len]

        final_scores = _weighted_relu_scores_triton(q_f32[b], k_tokens, w_f32[b])

        actual_topk = min(TOPK, seq_len)
        _, local_topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = local_topk_idx // PAGE_SIZE
        offset_per_token = local_topk_idx % PAGE_SIZE
        global_page_idx = page_indices[page_idx_per_token]
        token_idx = global_page_idx * PAGE_SIZE + offset_per_token
        topk_indices[b, :actual_topk] = token_idx.to(torch.int32)

    return (topk_indices,)


def kernel(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> Tuple[torch.Tensor]:
    return run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
