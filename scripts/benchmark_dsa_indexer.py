"""
Micro-benchmark for dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.

This script benchmarks the implementation in solution/triton/kernel.py with:
1) Triton score path (default)
2) Torch score path (set DSA_INDEXER_DISABLE_TRITON=1 internally)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from solution.triton.kernel import run as dsa_indexer_run

NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 64
HEAD_DIM_WITH_SCALE = 132
WORKLOAD_DEF = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"


def _load_workloads(dataset_root: Path, limit: int) -> List[Dict]:
    workload_path = dataset_root / "workloads" / "dsa_paged" / f"{WORKLOAD_DEF}.jsonl"
    rows: List[Dict] = []
    with workload_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if 0 < limit <= len(rows):
                break
    return rows


def _make_random_inputs(
    batch_size: int,
    num_pages: int,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_index_fp8 = (
        torch.randn((batch_size, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
    )

    k_fp8 = (
        torch.randn((num_pages, PAGE_SIZE, INDEX_HEAD_DIM), device=device, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
    )
    scales = torch.rand((num_pages, PAGE_SIZE, 1), device=device, dtype=torch.float32) + 1e-3

    k_u8 = k_fp8.view(torch.uint8).reshape(num_pages, PAGE_SIZE * INDEX_HEAD_DIM)
    scale_u8 = scales.view(torch.uint8).reshape(num_pages, PAGE_SIZE * 4)
    packed = torch.cat([k_u8, scale_u8], dim=1)
    k_index_cache_fp8 = packed.view(num_pages, PAGE_SIZE, 1, HEAD_DIM_WITH_SCALE).view(torch.int8)

    weights = torch.randn((batch_size, NUM_INDEX_HEADS), device=device, dtype=torch.float32)
    return (
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens.to(device),
        block_table.to(device),
    )


def _bench_once(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    warmup: int,
    iters: int,
    disable_triton: bool,
) -> Tuple[float, float, float]:
    if disable_triton:
        os.environ["DSA_INDEXER_DISABLE_TRITON"] = "1"
    else:
        os.environ.pop("DSA_INDEXER_DISABLE_TRITON", None)

    for _ in range(warmup):
        dsa_indexer_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        dsa_indexer_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
        ends[i].record()
    torch.cuda.synchronize()

    ms = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)], dtype=torch.float32)
    return float(ms.mean().item()), float(ms.quantile(0.5).item()), float(ms.quantile(0.9).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DSA topk indexer on CUDA GPU")
    parser.add_argument("--dataset-root", type=Path, default=Path("mlsys26-contest"))
    parser.add_argument("--num-workloads", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--torch-baseline", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Current PyTorch does not support float8_e4m3fn.")

    device = torch.device("cuda")
    rows = _load_workloads(args.dataset_root, args.num_workloads)
    if not rows:
        raise RuntimeError("No workloads loaded.")

    print(f"Loaded {len(rows)} workload(s) from {args.dataset_root}.")
    print("uuid batch max_pages seq_max triton_p50_ms triton_p90_ms torch_p50_ms torch_p90_ms exact_match")

    for row in rows:
        wl = row["workload"]
        uuid = wl["uuid"]
        axes = wl["axes"]
        batch_size = int(axes["batch_size"])
        num_pages = int(axes["num_pages"])
        max_num_pages = int(axes["max_num_pages"])

        tensor_file = args.dataset_root / wl["inputs"]["seq_lens"]["path"]
        packed = load_file(str(tensor_file))
        seq_lens = packed["seq_lens"].to(torch.int32)
        block_table = packed["block_table"].to(torch.int32)

        q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = _make_random_inputs(
            batch_size=batch_size,
            num_pages=num_pages,
            seq_lens=seq_lens,
            block_table=block_table,
            device=device,
        )

        _, tri_p50, tri_p90 = _bench_once(
            q_index_fp8,
            k_index_cache_fp8,
            weights,
            seq_lens,
            block_table,
            warmup=args.warmup,
            iters=args.iters,
            disable_triton=False,
        )

        torch_p50 = float("nan")
        torch_p90 = float("nan")
        exact_match = "na"
        if args.torch_baseline:
            os.environ.pop("DSA_INDEXER_DISABLE_TRITON", None)
            tri_out = dsa_indexer_run(
                q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table
            )[0]
            os.environ["DSA_INDEXER_DISABLE_TRITON"] = "1"
            ref_out = dsa_indexer_run(
                q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table
            )[0]
            exact_match = "yes" if torch.equal(tri_out, ref_out) else "no"

            _, torch_p50, torch_p90 = _bench_once(
                q_index_fp8,
                k_index_cache_fp8,
                weights,
                seq_lens,
                block_table,
                warmup=max(2, args.warmup // 2),
                iters=max(5, args.iters // 2),
                disable_triton=True,
            )

        seq_max = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
        print(
            f"{uuid[:8]} {batch_size:>3} {max_num_pages:>3} {seq_max:>5} "
            f"{tri_p50:>12.4f} {tri_p90:>12.4f} {torch_p50:>12.4f} {torch_p90:>12.4f} {exact_match}"
        )


if __name__ == "__main__":
    main()
