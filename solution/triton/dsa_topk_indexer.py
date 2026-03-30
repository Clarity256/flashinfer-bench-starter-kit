import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import triton
import triton.language as tl
from pathlib import Path

# -----------------------------------------------------------------------------
# FP8 dtype探测（适配不同Triton/架构）
# -----------------------------------------------------------------------------
def get_fp8_tl_dtype():
    for attr in ["float8e4m3fn", "float8e4nv", "float8e4b15"]:
        if hasattr(tl, attr):
            return getattr(tl, attr)
    return tl.float16  

FP8_TL = get_fp8_tl_dtype()

NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 64
TOPK = 2048
KV_CACHE_NUM_HEADS = 1
HEAD_DIM_WITH_SCALE = 132
PAD_PACKED_I64 = 0x007FFFFFFFFFFFFF  # ordered(-inf) << 32 | uint32(-1)

# -----------------------------------------------------------------------------
# float32 -> ordered uint32
# -----------------------------------------------------------------------------
@triton.jit
def f32_to_ordered_u32(x):
    u = x.to(tl.uint32, bitcast=True)
    sign = u >> 31
    # 修正：直接使用标量常量转换，避免显式指定 shape [1] 导致的广播错误
    xor_mask = tl.where(
        sign == 1,
        tl.cast(0xFFFFFFFF, tl.uint32),
        tl.cast(0x80000000, tl.uint32),
    )
    return u ^ xor_mask

# =============================================================================
# Kernel 1: 计算所有token的score并pack到uint64
# =============================================================================
@triton.jit
def dsa_pack_all_tokens_kernel(
    Q_ptr,                  
    K_u8_ptr,               
    W_ptr,                  
    Seq_Lens_ptr,           
    Block_Table_ptr,        
    Packed_Out_ptr,         
    stride_q_b, stride_q_h, stride_q_d,
    stride_w_b, stride_w_h,
    stride_bt_b, stride_bt_p,
    stride_out_b,
    NUM_HEADS: tl.constexpr,     
    HEAD_DIM: tl.constexpr,      
    PAGE_SIZE: tl.constexpr,     
    HEAD_DIM_WITH_SCALE: tl.constexpr,  
    BLOCK_PAGES: tl.constexpr,   
    MAX_NUM_PAGES: tl.constexpr, 
):
    pid_seg = tl.program_id(0)   
    pid_b   = tl.program_id(1)   

    L = tl.load(Seq_Lens_ptr + pid_b).to(tl.int32)
    active_pages = (L + PAGE_SIZE - 1) // PAGE_SIZE
    active_pages = tl.minimum(active_pages, tl.full([], MAX_NUM_PAGES, tl.int32))

    offs_h = tl.arange(0, NUM_HEADS)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + pid_b * stride_q_b + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
    q = tl.load(q_ptrs).to(FP8_TL)

    w_ptrs = W_ptr + pid_b * stride_w_b + offs_h * stride_w_h
    w = tl.load(w_ptrs).to(tl.float32)

    base_page = pid_seg * BLOCK_PAGES
    seg_out_base = pid_b * stride_out_b + pid_seg * (BLOCK_PAGES * PAGE_SIZE)

    offs_p = tl.arange(0, PAGE_SIZE)

    PAGE_STRIDE_BYTES = PAGE_SIZE * HEAD_DIM_WITH_SCALE     
    FP8_REGION_BYTES  = PAGE_SIZE * HEAD_DIM                

    for i in tl.static_range(0, BLOCK_PAGES):
        page_idx = base_page + i
        page_valid = page_idx < active_pages

        bt_ptr = Block_Table_ptr + pid_b * stride_bt_b + page_idx * stride_bt_p
        phys_page = tl.load(bt_ptr, mask=page_valid, other=-1).to(tl.int32)
        phys_valid = phys_page >= 0

        valid_count = L - page_idx * PAGE_SIZE
        tok_valid = offs_p < valid_count

        m_tok = page_valid & phys_valid & tok_valid   

        page_base = K_u8_ptr + phys_page * PAGE_STRIDE_BYTES  

        # Load as raw uint8 and bitcast to FP8 to avoid Triton casting `other=0` into FP8.
        k_u8_ptrs = page_base + offs_p[:, None] * HEAD_DIM + offs_d[None, :]
        k_u8_vals = tl.load(k_u8_ptrs, mask=m_tok[:, None], other=0)
        k = k_u8_vals.to(FP8_TL, bitcast=True)  

        scale_base = (page_base + FP8_REGION_BYTES).to(tl.pointer_type(tl.float32))
        scales = tl.load(scale_base + offs_p, mask=m_tok, other=0.0).to(tl.float32)  

        dots = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        dots = dots * scales[None, :]
        dots = tl.maximum(dots, 0.0)
        token_scores = tl.sum(dots * w[:, None], axis=0)  

        neg_inf = tl.full([PAGE_SIZE], -float("inf"), dtype=tl.float32)
        token_scores = tl.where(m_tok, token_scores, neg_inf)

        token_ids = phys_page * PAGE_SIZE + offs_p
        token_ids = tl.where(m_tok, token_ids, tl.full([PAGE_SIZE], -1, dtype=tl.int32))

        ordered = f32_to_ordered_u32(token_scores)  
        packed_u64 = (ordered.to(tl.uint64) << 32) | (token_ids.to(tl.uint32).to(tl.uint64))

        out_ptrs = (Packed_Out_ptr.to(tl.pointer_type(tl.uint64))  
                    + seg_out_base + i * PAGE_SIZE + offs_p)
        tl.store(out_ptrs, packed_u64)

# =============================================================================
# Kernel 2: 对packed全排序，取TopK indices
# =============================================================================
@triton.jit
def dsa_global_topk_from_packed_kernel(
    Packed_In_ptr,         
    Out_Indices_ptr,       
    stride_in_b,
    stride_out_b,
    SORT_LEN: tl.constexpr, # 强制为 2 的幂次方
    TOPK: tl.constexpr,
):
    pid_b = tl.program_id(0)

    offs = tl.arange(0, SORT_LEN)
    packed = tl.load(Packed_In_ptr.to(tl.pointer_type(tl.uint64)) + pid_b * stride_in_b + offs)

    sorted_packed = tl.sort(packed, descending=True)
    top_idx_all = (sorted_packed & tl.cast(0xFFFFFFFF, tl.uint64)).to(tl.int32)
    tl.store(Out_Indices_ptr + pid_b * stride_out_b + offs, top_idx_all, mask=offs < TOPK)

@triton.jit
def dsa_fill_packed_tail_kernel(
    Packed_Out_ptr,
    stride_out_b,
    start_idx,
    end_idx,
    PAD_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_blk = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs = start_idx + pid_blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < end_idx
    pad_vals = tl.full([BLOCK], PAD_VALUE, dtype=tl.uint64)
    out_ptrs = Packed_Out_ptr.to(tl.pointer_type(tl.uint64)) + pid_b * stride_out_b + offs
    tl.store(out_ptrs, pad_vals, mask=mask)

# -----------------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------------
def dsa_topk_indexer_fp8(
    q_index_fp8: torch.Tensor,          
    k_index_cache_fp8: torch.Tensor,    
    weights: torch.Tensor,              
    seq_lens: torch.Tensor,             
    block_table: torch.Tensor,          
    topk: int = TOPK,
    block_pages: int = 8,
    packed_workspace: torch.Tensor | None = None,
    out_workspace: torch.Tensor | None = None,
):
    assert q_index_fp8.is_cuda
    assert k_index_cache_fp8.is_cuda
    assert weights.is_cuda and seq_lens.is_cuda and block_table.is_cuda
    assert q_index_fp8.dtype == torch.float8_e4m3fn
    assert k_index_cache_fp8.dtype == torch.int8
    assert weights.dtype == torch.float32
    assert seq_lens.dtype == torch.int32
    assert block_table.dtype == torch.int32

    B, H, D = q_index_fp8.shape
    assert H == NUM_INDEX_HEADS and D == INDEX_HEAD_DIM
    assert weights.shape == (B, NUM_INDEX_HEADS)
    assert k_index_cache_fp8.shape[1] == PAGE_SIZE and k_index_cache_fp8.shape[2] == KV_CACHE_NUM_HEADS and k_index_cache_fp8.shape[3] == HEAD_DIM_WITH_SCALE

    max_num_pages = block_table.shape[1]
    assert topk == TOPK
    assert topk <= max_num_pages * PAGE_SIZE

    k_u8 = k_index_cache_fp8.view(torch.uint8)

    num_segs = (max_num_pages + block_pages - 1) // block_pages
    total_in = num_segs * block_pages * PAGE_SIZE
    
    # 修正：获取下一个 2 的幂次方，满足 Triton tl.arange 的硬性要求
    sort_len = 1 << (total_in - 1).bit_length()

    if packed_workspace is None:
        packed = torch.empty((B, sort_len), device=q_index_fp8.device, dtype=torch.int64)
    else:
        assert packed_workspace.is_cuda and packed_workspace.dtype == torch.int64
        assert packed_workspace.shape[0] == B and packed_workspace.shape[1] >= sort_len
        packed = packed_workspace[:, :sort_len]

    if out_workspace is None:
        out = torch.empty((B, topk), device=q_index_fp8.device, dtype=torch.int32)
    else:
        assert out_workspace.is_cuda and out_workspace.dtype == torch.int32
        assert out_workspace.shape[0] == B and out_workspace.shape[1] >= topk
        out = out_workspace[:, :topk]

    grid1 = (num_segs, B)
    dsa_pack_all_tokens_kernel[grid1](
        q_index_fp8, k_u8, weights, seq_lens, block_table, packed,
        q_index_fp8.stride(0), q_index_fp8.stride(1), q_index_fp8.stride(2),
        weights.stride(0), weights.stride(1),
        block_table.stride(0), block_table.stride(1),
        packed.stride(0),
        NUM_HEADS=NUM_INDEX_HEADS, HEAD_DIM=INDEX_HEAD_DIM, PAGE_SIZE=PAGE_SIZE,
        HEAD_DIM_WITH_SCALE=HEAD_DIM_WITH_SCALE,
        BLOCK_PAGES=block_pages,
        MAX_NUM_PAGES=max_num_pages,
        num_warps=4,
        num_stages=3,
    )

    if sort_len > total_in:
        tail_grid = (triton.cdiv(sort_len - total_in, 256), B)
        dsa_fill_packed_tail_kernel[tail_grid](
            packed,
            packed.stride(0),
            total_in,
            sort_len,
            PAD_VALUE=PAD_PACKED_I64,
            BLOCK=256,
            num_warps=4,
            num_stages=1,
        )

    grid2 = (B,)
    dsa_global_topk_from_packed_kernel[grid2](
        packed, out,
        packed.stride(0), out.stride(0),
        SORT_LEN=sort_len,
        TOPK=topk,
        num_warps=8,
        num_stages=2,
    )
    return out

def dsa_topk_indexer_fp8_h64_d128_topk2048_ps64(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
):
    return dsa_topk_indexer_fp8(
        q_index_fp8=q_index_fp8,
        k_index_cache_fp8=k_index_cache_fp8,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
        topk=TOPK,
        block_pages=8,
    )

# -----------------------------------------------------------------------------
# 生成符合deep_gemm pack的k_index_cache_fp8（用于自测）
# -----------------------------------------------------------------------------
def make_packed_k_cache(num_pages: int, device: str = "cuda"):
    PAGE_SIZE = 64
    HEAD_DIM = 128
    HEAD_DIM_WITH_SCALE = 132

    fp16 = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM), device=device, dtype=torch.float16)
    if hasattr(torch, "float8_e4m3fn"):
        fp8 = fp16.to(torch.float8_e4m3fn)
        fp8_u8 = fp8.view(torch.uint8)
    else:
        fp8_u8 = fp16.view(torch.uint8)[:, :, :HEAD_DIM]

    scale = torch.rand((num_pages, PAGE_SIZE), device=device, dtype=torch.float32) * 2.0

    kv_flat = torch.empty((num_pages, PAGE_SIZE * HEAD_DIM_WITH_SCALE), device=device, dtype=torch.uint8)
    kv_flat[:, :PAGE_SIZE * HEAD_DIM] = fp8_u8.reshape(num_pages, PAGE_SIZE * HEAD_DIM)
    kv_flat[:, PAGE_SIZE * HEAD_DIM:] = scale.view(torch.uint8).reshape(num_pages, PAGE_SIZE * 4)

    k_int8 = kv_flat.view(torch.int8).view(num_pages, PAGE_SIZE, 1, HEAD_DIM_WITH_SCALE)
    return k_int8

def benchmark_and_profile(trace_file: str = "json/2.24indexer_1_profile.json"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for profiling.")

    device = "cuda"
    B = 8
    MAX_SEQ_LEN = 4096
    max_num_pages = (MAX_SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE

    q_fp16 = torch.randn((B, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float16)
    if hasattr(torch, "float8_e4m3fn"):
        q = q_fp16.to(torch.float8_e4m3fn)
    else:
        q = q_fp16

    weights = torch.randn((B, NUM_INDEX_HEADS), device=device, dtype=torch.float32)
    seq_lens = torch.randint(1, MAX_SEQ_LEN + 1, (B,), device=device, dtype=torch.int32)

    block_table = torch.full((B, max_num_pages), -1, device=device, dtype=torch.int32)
    for b in range(B):
        n = (int(seq_lens[b].item()) + PAGE_SIZE - 1) // PAGE_SIZE
        block_table[b, :n] = torch.arange(0, n, device=device, dtype=torch.int32)

    k_cache = make_packed_k_cache(num_pages=max_num_pages, device=device)

    for _ in range(5):
        _ = dsa_topk_indexer_fp8_h64_d128_topk2048_ps64(q, k_cache, weights, seq_lens, block_table)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iters = 50
    start.record()
    for _ in range(iters):
        _ = dsa_topk_indexer_fp8_h64_d128_topk2048_ps64(q, k_cache, weights, seq_lens, block_table)
    end.record()
    torch.cuda.synchronize()
    print(f"avg latency: {start.elapsed_time(end) / iters:.4f} ms")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            _ = dsa_topk_indexer_fp8_h64_d128_topk2048_ps64(q, k_cache, weights, seq_lens, block_table)
            prof.step()

    trace_path = Path(trace_file)
    if not trace_path.is_absolute():
        trace_path = Path(__file__).resolve().parent / trace_path
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    prof.export_chrome_trace(str(trace_path))
    print(f"trace saved to {trace_path}")

def main():
    benchmark_and_profile()

if __name__ == "__main__":
    main()