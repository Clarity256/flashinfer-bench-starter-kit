"""
Thin entry-point wrapper for the DSA top-k indexer implementation.

`config.toml` points to `kernel`, while the real implementation lives in
`dsa_topk_indexer.py` under the definition-specific function name.
"""

from dsa_topk_indexer import dsa_topk_indexer_fp8_h64_d128_topk2048_ps64


def kernel(
    q_index_fp8,
    k_index_cache_fp8,
    weights,
    seq_lens,
    block_table,
):
    return dsa_topk_indexer_fp8_h64_d128_topk2048_ps64(
        q_index_fp8=q_index_fp8,
        k_index_cache_fp8=k_index_cache_fp8,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
    )
