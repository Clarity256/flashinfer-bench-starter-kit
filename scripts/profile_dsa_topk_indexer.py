"""
Minimal standalone driver for the Triton DSA top-k indexer.

Use this when you want a stable process entry for Nsight Compute profiling
without going through the full flashinfer_bench runner.
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "solution" / "triton"))

from dsa_topk_indexer import dsa_topk_indexer_fp8_h64_d128_topk2048_ps64


def make_inputs(
    batch_size: int,
    max_num_pages: int,
    num_pages: int,
    seed: int,
    device: str,
):
    page_size = 64
    num_index_heads = 64
    index_head_dim = 128
    head_dim_with_scale = 132

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    q_fp16 = torch.randn(
        (batch_size, num_index_heads, index_head_dim),
        device=device,
        dtype=torch.float16,
        generator=g,
    )
    q_index_fp8 = q_fp16.to(torch.float8_e4m3fn)

    weights = torch.randn(
        (batch_size, num_index_heads),
        device=device,
        dtype=torch.float32,
        generator=g,
    )

    seq_lens = torch.randint(
        low=1,
        high=max_num_pages * page_size + 1,
        size=(batch_size,),
        device=device,
        dtype=torch.int32,
        generator=g,
    )

    block_table = torch.full(
        (batch_size, max_num_pages),
        fill_value=-1,
        device=device,
        dtype=torch.int32,
    )
    for b in range(batch_size):
        n = (int(seq_lens[b].item()) + page_size - 1) // page_size
        perm = torch.randperm(num_pages, device=device, generator=g)[:n]
        block_table[b, :n] = perm.to(torch.int32)

    fp16 = torch.randn(
        (num_pages, page_size, index_head_dim),
        device=device,
        dtype=torch.float16,
        generator=g,
    )
    fp8 = fp16.to(torch.float8_e4m3fn)
    fp8_u8 = fp8.view(torch.uint8)
    scale = torch.rand(
        (num_pages, page_size),
        device=device,
        dtype=torch.float32,
        generator=g,
    ) * 2.0

    kv_flat = torch.empty(
        (num_pages, page_size * head_dim_with_scale),
        device=device,
        dtype=torch.uint8,
    )
    kv_flat[:, : page_size * index_head_dim] = fp8_u8.reshape(num_pages, page_size * index_head_dim)
    kv_flat[:, page_size * index_head_dim :] = scale.view(torch.uint8).reshape(num_pages, page_size * 4)
    k_index_cache_fp8 = kv_flat.view(torch.int8).view(num_pages, page_size, 1, head_dim_with_scale)

    return q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table


def main():
    parser = argparse.ArgumentParser(description="Profile the Triton DSA top-k indexer.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-num-pages", type=int, default=32)
    parser.add_argument("--num-pages", type=int, default=11923)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required.")

    inputs = make_inputs(
        batch_size=args.batch_size,
        max_num_pages=args.max_num_pages,
        num_pages=args.num_pages,
        seed=args.seed,
        device=args.device,
    )

    for _ in range(args.warmup):
        _ = dsa_topk_indexer_fp8_h64_d128_topk2048_ps64(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        out = dsa_topk_indexer_fp8_h64_d128_topk2048_ps64(*inputs)
    end.record()
    torch.cuda.synchronize()

    print(f"output shape: {tuple(out.shape)}")
    print(f"avg latency: {start.elapsed_time(end) / args.iters:.4f} ms")


if __name__ == "__main__":
    main()
