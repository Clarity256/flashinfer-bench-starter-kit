# DSA TopK Indexer 文档（脚本说明 + 设计方法）

本文档整理以下内容：

1. `scripts/benchmark_dsa_indexer.py` 的功能与使用方式  
2. `scripts/profile_dsa_indexer_b200.sh` 的功能与使用方式  
3. `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` 的设计方法（含当前 Triton 实现）

---

## 1. `benchmark_dsa_indexer.py`

### 1.1 脚本目标

`scripts/benchmark_dsa_indexer.py` 是一个针对 `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` 的微基准脚本，核心用途是：

1. 批量读取 workload（来自 `mlsys26-contest/workloads/dsa_paged/*.jsonl`）  
2. 生成与定义一致的随机输入（FP8 query、FP8 KV cache、weights）  
3. 运行 `solution/triton/kernel.py::run` 并测量延迟  
4. 可选地与 Torch 评分路径做对比，输出 `exact_match`

### 1.2 主要流程

脚本在每个 workload 上执行如下步骤：

1. 读取 workload 的 axes（`batch_size`、`num_pages`、`max_num_pages`）  
2. 从 safetensors 文件读取 `seq_lens` 和 `block_table`  
3. 构造随机输入：
   - `q_index_fp8`: `[B, 64, 128]`，`float8_e4m3fn`
   - `k_index_cache_fp8`: `[num_pages, 64, 1, 132]`，`int8`（按 DeepGEMM 布局打包）
   - `weights`: `[B, 64]`，`float32`
4. 先 warmup，再使用 CUDA Event 统计 `iters` 次延迟  
5. 输出每个 workload 的 p50/p90（以及可选 Torch baseline）

### 1.3 参数说明

```bash
python scripts/benchmark_dsa_indexer.py \
  --dataset-root mlsys26-contest \
  --num-workloads 20 \
  --warmup 20 \
  --iters 100 \
  --torch-baseline
```

- `--dataset-root`: TraceSet 根目录，默认 `mlsys26-contest`
- `--num-workloads`: 读取 workload 数量，默认 `5`
- `--warmup`: 每个 workload 的预热轮数，默认 `10`
- `--iters`: 每个 workload 的计时轮数，默认 `50`
- `--torch-baseline`: 开启后会额外跑 Torch 路径，并输出 `exact_match`

### 1.4 输出列解释

运行后会打印如下列：

- `uuid`: workload UUID 前缀
- `batch`: batch size
- `max_pages`: max_num_pages
- `seq_max`: 该 workload 中 `seq_lens.max()`
- `triton_p50_ms` / `triton_p90_ms`: Triton 路径延迟
- `torch_p50_ms` / `torch_p90_ms`: Torch 评分路径延迟（未开启 baseline 时为 `nan`）
- `exact_match`: Triton 输出与 Torch 输出是否逐元素一致（`yes/no/na`）

### 1.5 环境变量

- `DSA_INDEXER_DISABLE_TRITON=1`
  - 作用：禁用 Triton 评分核，强制走 Torch 评分路径
  - 用途：快速做 correctness 对照

示例：

```bash
DSA_INDEXER_DISABLE_TRITON=1 \
python scripts/benchmark_dsa_indexer.py --dataset-root mlsys26-contest --num-workloads 2
```

### 1.6 适用场景与注意事项

1. 该脚本是微基准，不等同于 FlashInfer-Bench 全流程评测。  
2. 输入中的 `q/k/weights` 为随机生成，`seq_lens/block_table` 来自真实 workload。  
3. 若出现 `exact_match=no`，通常表示两条评分路径在浮点累加细节上存在差异，未必是形状/索引错误。

---

## 2. `profile_dsa_indexer_b200.sh`

### 2.1 脚本目标

`scripts/profile_dsa_indexer_b200.sh` 是一键 profiling 脚本，串行执行三件事：

1. 延迟对比（Triton vs Torch）  
2. NSYS 时间线采集  
3. NCU 指标采集

### 2.2 使用方式

```bash
bash scripts/profile_dsa_indexer_b200.sh [DATASET_ROOT] [OUT_DIR]
```

默认值：

- `DATASET_ROOT=mlsys26-contest`
- `OUT_DIR=profile_out`

示例：

```bash
bash scripts/profile_dsa_indexer_b200.sh mlsys26-contest profile_out_5090
```

### 2.3 三个阶段说明

1. **阶段 1：延迟对比**  
   调用 `benchmark_dsa_indexer.py --torch-baseline`，并保存到 `OUT_DIR/latency.txt`。

2. **阶段 2：NSYS**  
   采集 CUDA/NVTX/OS Runtime 时间线，输出 `OUT_DIR/nsys_dsa_indexer.*`。

3. **阶段 3：NCU**  
   对 `weighted_relu_score_kernel` 进行 full set 指标采集，输出 `OUT_DIR/ncu_dsa_indexer.csv`。

### 2.4 结果文件

典型输出目录结构：

```text
profile_out/
├── latency.txt
├── nsys_dsa_indexer.nsys-rep
└── ncu_dsa_indexer.csv
```

### 2.5 运行前置条件

1. `nsys` 与 `ncu` 在 PATH 中可用  
2. CUDA 驱动与 Toolkit 匹配  
3. Python 环境已安装 `torch/triton/safetensors`

---

## 3. `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` 设计方法

### 3.1 算子定义摘要

目标定义：`mlsys26-contest/definitions/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json`

关键约束：

1. `num_index_heads = 64`
2. `index_head_dim = 128`
3. `page_size = 64`
4. `topk = 2048`
5. `k_index_cache_fp8` 的最后一维是 `132 = 128(fp8) + 4(scale bytes)`

数学形式：

```text
final_score[t] = sum_h relu(dot(q[h], k[t])) * weights[h]
topk_indices = topk(final_score, k=2048) 映射到全局 token id
```

### 3.2 数据布局与解码策略

`k_index_cache_fp8` 的真实页内布局为：

1. 前 `64 * 128` 字节：所有 token 的 FP8 向量数据
2. 后 `64 * 4` 字节：每个 token 的缩放因子（float32 字节）

实现中采用如下解码路径：

1. `int8 -> uint8` 视图
2. 按页拉平为 `[num_pages, 64 * 132]`
3. 前段视图为 `float8_e4m3fn` 并转 `float32`
4. 后段视图为 `float32` scale
5. `dequant_k = fp8_float * scale`

### 3.3 当前实现架构

文件：`solution/triton/kernel.py`

当前 `run(...)` 的分层：

1. **输入校验**  
   检查 rank、常量维度、batch 对齐。

2. **按 batch 处理**  
   对每个样本，根据 `seq_len` 与 `block_table` 取出对应页。

3. **页解码 + token 展平**  
   将选中页解码成 `[num_pages_for_seq, 64, 128]`，再展平截断为 `[seq_len, 128]`。

4. **评分计算**  
   默认调用 Triton 核 `_weighted_relu_score_kernel`；可通过环境变量切回 Torch。

5. **TopK 与全局索引映射**  
   `local_topk_idx -> (page_idx, offset) -> global_token_idx`，并写入输出。

### 3.4 Triton 核设计点

评分核的核心是分块 GEMM 风格计算：

1. 程序维度：按 token block 切分（`BLOCK_TOKENS`）  
2. 头维分块：`BLOCK_HEADS`（当前配置为 `16/32`）  
3. dim 分块：`BLOCK_D=32`  
4. 在每个 head block 内：
   - 累积 `dot(q_tile, k_tile)`（`input_precision="ieee"`）
   - ReLU
   - 乘 `weights`
   - 沿 head 规约到 token 分数

autotune 配置当前为：

1. `(BLOCK_TOKENS=64,  BLOCK_HEADS=16, BLOCK_D=32)`
2. `(BLOCK_TOKENS=128, BLOCK_HEADS=16, BLOCK_D=32)`
3. `(BLOCK_TOKENS=256, BLOCK_HEADS=16, BLOCK_D=32)`
4. `(BLOCK_TOKENS=128, BLOCK_HEADS=32, BLOCK_D=32)`

### 3.5 正确性与性能迭代建议

从工程角度，建议按以下顺序迭代：

1. **先 correctness**  
   在小 workload 上确保索引映射和页解码完全正确。

2. **再比对数值一致性**  
   用 `--torch-baseline` 观察 `exact_match`，定位浮点路径差异。

3. **再做性能优化**  
   通过 `profile_dsa_indexer_b200.sh` 收集 NCU/NSYS 证据后再调参。

4. **最后做 Blackwell 专项优化**  
   围绕占用率、L2 命中、指令吞吐、访存模式优化 tile/warp/stage。

### 3.6 常见问题

1. **`ModuleNotFoundError: No module named 'solution'`**  
   直接在 `scripts/` 下运行时发生。脚本已加入 `PROJECT_ROOT` 到 `sys.path`。

2. **Triton 编译报 global variable 非 constexpr**  
   需将常量改为 kernel meta 参数（`tl.constexpr`）并在 launch 显式传入。

3. **`tl.dot` 维度断言（M/N/K >= 16）**  
   需避免 `BLOCK_HEADS < 16`。

4. **`exact_match=no`**  
   通常是浮点路径差异引起，先确认功能正确，再决定是优先精度一致还是优先速度。

---

## 4. 快速命令清单

### 4.1 跑微基准（含 Torch 对照）

```bash
python scripts/benchmark_dsa_indexer.py \
  --dataset-root mlsys26-contest \
  --num-workloads 20 \
  --warmup 20 \
  --iters 100 \
  --torch-baseline
```

### 4.2 跑一键 profiling

```bash
bash scripts/profile_dsa_indexer_b200.sh mlsys26-contest profile_out
```

### 4.3 强制 Torch 路径做核对

```bash
DSA_INDEXER_DISABLE_TRITON=1 \
python scripts/benchmark_dsa_indexer.py --dataset-root mlsys26-contest --num-workloads 5
```

