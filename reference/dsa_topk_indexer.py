import torch


def dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    """
    把 deep_gemm 风格打包的 FP8 K cache 反量化成 float32。

    输入:
        k_index_cache_fp8:
            shape = [num_pages, page_size, 1, 132]
            dtype = int8
            但里面存的是“字节”，真正解释时应按 uint8 看

    物理布局（每个 page）:
        [所有 token 的 fp8 数据][所有 token 的 scale]
        也就是:
        - 前 page_size * 128 字节: fp8 key 数据
        - 后 page_size * 4 字节:  每个 token 一个 float32 scale

    输出:
        K_all:
            shape = [num_pages, page_size, 128]
            dtype = float32
    """
    # 1) int8 只是“装字节的壳”，真正要按 uint8 来解释
    k_bytes = k_index_cache_fp8.view(torch.uint8)

    num_pages, page_size, kv_cache_num_heads, head_dim_with_scale = k_bytes.shape
    assert kv_cache_num_heads == 1, "这个 reference 假设 KV cache 只有 1 个 head"
    assert head_dim_with_scale == 132, "这里固定是 128 维 fp8 数据 + 4 字节 scale"

    head_dim = 128

    # 2) 每个 page 先拉平成一维，方便按“前半段 fp8 / 后半段 scale”切开
    kv_flat = k_bytes.reshape(num_pages, page_size * head_dim_with_scale)

    # 前 page_size * 128 字节是 fp8 数据
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()

    # 后 page_size * 4 字节是 scale
    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()

    # 3) 重新解释 fp8 数据
    #    这里每个元素 1 字节，所以可以直接 view 成 float8_e4m3fn
    fp8_tensor = fp8_bytes.view(torch.float8_e4m3fn).reshape(num_pages, page_size, head_dim)

    # 转成 float32，方便后面计算
    fp8_float = fp8_tensor.to(torch.float32)

    # 4) scale 部分原来是按字节存的 float32
    #    现在把它重新解释成 float32，再 reshape 成 [num_pages, page_size, 1]
    scale = scale_bytes.view(torch.float32).reshape(num_pages, page_size, 1)

    # 5) 真正反量化
    K_all = fp8_float * scale
    return K_all


@torch.no_grad()
def run(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
):
    """
    Reference 实现。

    输入:
        q_index_fp8:      [batch_size, 64, 128], float8_e4m3fn
        k_index_cache_fp8:[num_pages, 64, 1, 132], int8(按 uint8 解读)
        weights:          [batch_size, 64], float32
        seq_lens:         [batch_size], int32
        block_table:      [batch_size, max_num_pages], int32

    输出:
        topk_indices:     [batch_size, 2048], int32
    """
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, head_dim_with_scale = k_index_cache_fp8.shape
    topk = 2048

    # 固定常量检查
    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64
    assert head_dim_with_scale == 132

    device = q_index_fp8.device

    # query 直接转 float32
    q = q_index_fp8.to(torch.float32)  # [batch_size, 64, 128]

    # 整个 K cache 先反量化
    # 这是 reference 写法，清晰但不高效
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages, 64, 128]

    # 默认全填 -1，表示 padding
    topk_indices = torch.full(
        (batch_size, topk),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )

    max_num_pages = block_table.shape[1]

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())

        # 空序列直接跳过
        if seq_len == 0:
            continue

        # 这个样本实际用了多少个 page
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        assert num_pages_for_seq <= max_num_pages

        # 取出该样本对应的物理 page 编号
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        # Gather 这些 page 的 key
        # [num_pages_for_seq, 64, 128]
        K_paged = K_all[page_indices]

        # 拉平成 token 序列，然后截断到真实 seq_len
        # [num_pages_for_seq * 64, 128] -> [seq_len, 128]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]

        # 当前 batch 的 query: [64, 128]
        q_b = q[b]

        # 点积打分
        # [64, 128] @ [128, seq_len] -> [64, seq_len]
        scores = q_b @ K.T

        # ReLU
        scores_relu = torch.relu(scores)

        # head 权重
        w = weights[b]  # [64]

        # 每个 head 乘权重，再沿 head 维求和
        # [64, seq_len] * [64, 1] -> [64, seq_len] -> [seq_len]
        final_scores = (scores_relu * w[:, None]).sum(dim=0)

        # 实际 topk 不能超过序列长度
        actual_topk = min(topk, seq_len)

        # 取前 actual_topk 个分数最高的位置
        _, topk_idx = torch.topk(final_scores, k=actual_topk, dim=0)

        # topk_idx 现在是“序列内部位置”
        # 需要变成“全局物理 token index”
        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size

        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)

def pack_fp8_kv_cache_for_test(K_float: torch.Tensor):
    """
    仅用于测试:
    把 [num_pages, page_size, 128] 的 float32 K
    打包成 deep_gemm 风格的 [num_pages, page_size, 1, 132] int8

    返回:
        packed_cache, scale
    """
    num_pages, page_size, head_dim = K_float.shape
    assert head_dim == 128

    # 给每个 token 一个 scale
    # 这里不是官方唯一做法，只是为了构造一个可运行测试
    scale = K_float.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0

    # 量化到 FP8 前先除 scale
    K_scaled = K_float / scale
    K_fp8 = K_scaled.to(torch.float8_e4m3fn)

    # 取出 fp8 字节
    fp8_bytes = K_fp8.contiguous().view(torch.uint8).reshape(num_pages, page_size * head_dim)

    # 取出 scale 的原始字节
    scale_bytes = scale.contiguous().view(torch.uint8).reshape(num_pages, page_size * 4)

    # 按 deep_gemm 布局拼起来: [fp8 数据][scale 数据]
    packed = torch.cat([fp8_bytes, scale_bytes], dim=1)

    # reshape 回 [num_pages, page_size, 1, 132]
    packed = packed.reshape(num_pages, page_size, 1, 132)

    # 外部规格要求 dtype 看起来是 int8
    packed = packed.view(torch.int8)

    return packed, scale

def demo():
    device = "cuda"

    batch_size = 2
    num_pages = 6
    page_size = 64
    num_heads = 64
    head_dim = 128
    topk = 2048
    max_num_pages = 4

    # 构造 query
    q_float = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)
    q_index_fp8 = q_float.to(torch.float8_e4m3fn)

    # 构造 K cache
    K_float = torch.randn(num_pages, page_size, head_dim, device=device, dtype=torch.float32)
    k_index_cache_fp8, _ = pack_fp8_kv_cache_for_test(K_float)

    # 构造 weights
    weights = torch.randn(batch_size, num_heads, device=device, dtype=torch.float32)

    # 两个样本的长度
    seq_lens = torch.tensor([100, 150], device=device, dtype=torch.int32)

    # block_table:
    # 第 0 个样本用 page 2, 4
    # 第 1 个样本用 page 1, 3, 5
    block_table = torch.tensor(
        [
            [2, 4, 0, 0],
            [1, 3, 5, 0],
        ],
        device=device,
        dtype=torch.int32,
    )

    topk_indices, = run(
        q_index_fp8=q_index_fp8,
        k_index_cache_fp8=k_index_cache_fp8,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
    )

    print("topk_indices.shape =", topk_indices.shape)
    print("sample 0 first 20 indices =", topk_indices[0, :20])
    print("sample 1 first 20 indices =", topk_indices[1, :20])


if __name__ == "__main__":
    demo()