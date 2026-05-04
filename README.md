# SQL2AST 数据处理与训练指南

本文档面向 Linux 云服务器环境，按“数据准备 -> 候选 SQL 执行评估 -> AST/距离处理 -> SFT -> DPO”的顺序说明完整流程。示例默认运行环境为 8 张 A100 40G，训练阶段建议使用多卡启动。

## 1. 项目功能概览

仓库中主要包含 5 类脚本：

- `eval.py`
  将候选 SQL 在对应 SQLite 数据库上执行，并与 gold SQL 的执行结果比较，输出 `correct_set` 和 `wrong_set`。
- `sql_to_ast.py`
  将 `eval.py` 的输出转换为规范化 SQL / AST 结果，方便分析与调试。
- `src/calibrate.py`
  统计 AST 距离与执行正确性的相关性，用于检查距离函数是否可靠。
- `src/build_pairs.py`
  基于 `eval_results` 构造 DPO 所需的偏好数据 `dpo_pairs.jsonl`。
- `src/filter_dpo_pairs.py`
  对全量 DPO pair 做去重、长度过滤和按 prompt 截断，生成可直接训练的 DPO 文件。
- `src/build_cscsql_contents_index.py`
  为 `csc_sql` 风格 prompt 构造建立每个数据库的内容检索索引。
- `src/train_sft.py` / `src/train_dpo.py`
  分别进行 SFT 训练和 DPO 训练。

其中：

- `eval.py` 决定候选 SQL “执行上对不对”
- `src/distance/*` 决定候选 SQL “结构上离 gold 有多近”
- `src/train_sft.py` 先让模型学会基础 NL2SQL
- `src/train_dpo.py` 再让模型偏向更优候选

## 2. 推荐运行环境

### 2.1 硬件

- GPU: 8 x A100 40G
- CPU: 建议 32 核及以上
- 内存: 建议 128 GB 及以上
- 磁盘: 建议至少 500 GB 可用空间

### 2.2 软件

- Linux x86_64
- Python 3.10 或 3.11
- CUDA 11.8 或 12.x
- 建议使用 `conda` 或 `venv`

### 2.3 Python 依赖

仓库中的 `requirements.txt` 目前只包含：

```bash
pip install -r requirements.txt
```

这只会安装 `sqlglot`，足够运行 AST 相关脚本，但不足以直接训练。若要完整跑通 SFT / DPO，建议额外安装：

```bash
pip install pyyaml datasets transformers peft trl accelerate sentencepiece
pip install scipy scikit-learn
```

再根据你的 CUDA 版本安装 PyTorch。以 CUDA 11.8 为例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果服务器环境由平台统一维护，也可以使用平台预装的 PyTorch。

如果你使用当前仓库默认的 `csc_sql` 风格 prompt（`CREATE TABLE ...` + sampled values + question-related DB values），还需要补充：

```bash
pip install pyserini pyjnius nltk ijson func_timeout
```

并安装可用的 Java / `javac`。实践中建议使用 OpenJDK 21。

## 3. 数据目录约定

建议在服务器上准备如下目录结构：

```text
/data/sql2ast/
├── train/
│   ├── train.json
│   └── train_databases/
│       ├── db1.sqlite
│       └── db2/
│           └── db2.sqlite
├── candidates/
│   ├── 0_xxx.json
│   ├── 1_xxx.json
│   └── ...
├── eval_results/
├── eval_results_ast/
├── reports/
└── outputs/
    ├── sft/
    └── dpo/
```

### 3.1 `train.json`

训练集中的每条样本至少应包含这些字段：

- `question`
- `db_id`
- `SQL`
- `evidence`

其中：

- `question` 是自然语言问题
- `db_id` 是数据库标识
- `SQL` 是 gold SQL
- `evidence` 可以为空字符串

### 3.2 候选 SQL 文件

每个候选文件名必须能从前缀提取 `sample_id`，例如：

```text
544_movies_4.json
```

文件内容应为 JSON 列表，每条记录里包含 `all_sqls`：

```json
[
  {
    "all_sqls": [
      "SELECT ...",
      "SELECT ..."
    ]
  }
]
```

`eval.py` 会从文件名里取出 `544`，并默认使用 `train.json[544]` 作为对应样本。

## 4. 配置 `.location`

在仓库根目录创建 `.location`：

```ini
TRAIN_DATA_PATH=/data/sql2ast/train/train.json
TRAIN_DATABASE_PATH=/data/sql2ast/train/train_databases
SQL_PATH=/data/sql2ast/candidates
EVAL_OUTPUT_PATH=/data/sql2ast/eval_results
```

字段说明：

- `TRAIN_DATA_PATH`
  训练集 JSON 路径
- `TRAIN_DATABASE_PATH`
  SQLite 数据库根目录
- `SQL_PATH`
  候选 SQL 文件或目录
- `EVAL_OUTPUT_PATH`
  `eval.py` 的输出目录，可选，不填时默认 `eval_results`

## 5. 完整流程总览

完整流程建议按下面顺序执行：

```text
1. 安装依赖
2. 配置 .location
3. 执行 eval.py，得到 eval_results
4. 执行 sql_to_ast.py，得到 eval_results_ast
5. 执行 calibrate.py，检查距离函数质量
6. 执行 build_pairs.py，生成 dpo_pairs.jsonl
7. 执行 train_sft.py，得到 outputs/sft
8. 执行 train_dpo.py，得到 outputs/dpo
9. 对 SFT / DPO 模型分别做推理与评估
```

### 5.1 当前推荐的 CHES 训练数据工作流（2026-05）

如果你当前跑的是 CHES 这套数据，而不是完全从零搭目录，建议采用下面这套更贴近当前实验状态的工作流：

1. 先为训练库建立 `csc_sql` prompt 所需的 DB 内容索引：

```bash
python src/build_cscsql_contents_index.py \
  --database-root /workspace/data/ches/train_databases
```

默认输出到：

```text
/workspace/data/ches/train_db_contents_index
```

2. `SFT` 训练数据推荐直接使用：

```text
/workspace/data/ches/data/20260428_refresh/sft_augmented.recommended.json
```

这份文件是“原始 `train.json` + 执行正确 sampled SQL 增广”的推荐集合。  
`SFT` prompt 在训练时动态构造，所以 **不需要** 重写 JSON；只要索引准备好，训练时就会自动使用 richer prompt。

3. `DPO` 推荐先生成 / 使用 fully indexed 的全量 pair 文件，再过滤为最终训练集。当前推荐的最终训练文件是：

```text
/workspace/data/ches/data/20260428_refresh/dpo_pairs.strict_mix.cscsql_prompt.2048_2560.jsonl
```

它的过滤规则是：

- `allowed_pair_types = correct_wrong,wrong_wrong`
- `min_margin = 0.05`
- `max_prompt_tokens = 2048`
- `max_total_tokens = 2560`
- `max_pairs_per_prompt = 4`

当前统计：

- total pairs: `18,700`
- `correct_wrong`: `10,042`
- `wrong_wrong`: `8,658`
- unique prompts: `4,935`

如果你更关心 prompt 覆盖，`min_margin = 0.03` 也试过，但相比 `0.05` 只多出 `336` 对和 `63` 个 prompt，增益不大，因此当前仍推荐 `0.05`。

4. 如果你要继续做新的 CHES 实验，优先顺序建议是：

- 先建完 `train_db_contents_index`
- 再重写 `dpo_pairs.with_meta.full.cscsql_prompt.jsonl`
- 再过滤出最终训练文件
- 然后重跑 `SFT` / `DPO`

下面按阶段展开。

## 6. 阶段一：候选 SQL 执行评估

### 6.1 运行命令

最常用命令：

```bash
python eval.py --output-dir /data/sql2ast/eval_results
```

常见可选参数：

```bash
python eval.py --limit 5 --output-dir /data/sql2ast/eval_results
python eval.py --ignore-order --output-dir /data/sql2ast/eval_results
python eval.py --num-cpus 8 --output-dir /data/sql2ast/eval_results
```

参数说明：

- `--limit`
  只处理前 N 个文件，适合冒烟测试
- `--ignore-order`
  比较结果集时忽略行顺序
- `--num-cpus`
  并行执行候选 SQL 的 CPU worker 数

### 6.2 输出内容

每个样本会生成一个 `*_eval.json`，例如：

```text
/data/sql2ast/eval_results/544_movies_4_eval.json
```

同时会生成汇总文件：

```text
/data/sql2ast/eval_results/summary.json
```

单个 `*_eval.json` 里主要包含：

- `correct_set`
  执行结果与 gold SQL 一致的候选
- `wrong_set`
  执行结果与 gold SQL 不一致的候选
- `metadata`
  样本级元信息
- `file_errors`
  文件级报错

这一步是后续所有 AST 分析、pair 构造、DPO 训练的基础。

## 7. 阶段二：SQL 规范化与 AST 导出

### 7.1 运行命令

将整个 `eval_results` 目录转换为 AST 输出：

```bash
python sql_to_ast.py /data/sql2ast/eval_results \
  -o /data/sql2ast/eval_results_ast \
  --pattern '*_eval.json' \
  --dialect sqlite
```

如果希望对每个集合内部先去重：

```bash
python sql_to_ast.py /data/sql2ast/eval_results \
  -o /data/sql2ast/eval_results_ast \
  --pattern '*_eval.json' \
  --dialect sqlite \
  --deduplicate
```

### 7.2 输出内容

会生成例如：

```text
/data/sql2ast/eval_results_ast/544_movies_4_ast.json
```

文件中会包含：

- `gold`
- `correct`
- `wrong`
- 每条 SQL 对应的 `normalized_sql`
- 解析失败信息 `errors`

这一步主要用于：

- 验证 SQL 是否可被 `sqlglot` 解析
- 统一 SQL 表达形式
- 为距离函数和误差分析提供更稳定输入

## 8. 阶段三：校准距离函数

这一步不是必须，但强烈建议在正式构造 DPO 数据前先执行。

### 8.1 运行命令

```bash
python src/calibrate.py \
  --eval-dir /data/sql2ast/eval_results \
  --train-data /data/sql2ast/train/train.json \
  --database-root /data/sql2ast/train/train_databases \
  --output /data/sql2ast/reports/distance_calibration.json
```

### 8.2 作用

该脚本会统计：

- AST 距离与执行正确性的 Spearman 相关性
- 正样本和负样本的平均距离
- AUC 等指标

如果距离函数与执行正确性几乎没有相关性，那么基于该距离构造的 DPO 数据质量会受影响。

## 9. 阶段四：构造 DPO 偏好数据

### 9.1 运行命令

```bash
python src/build_pairs.py \
  --eval-dir /data/sql2ast/eval_results \
  --train-data /data/sql2ast/train/train.json \
  --database-root /data/sql2ast/train/train_databases \
  --output /data/sql2ast/data/dpo_pairs.jsonl
```

如果只想先跑少量样本：

```bash
python src/build_pairs.py \
  --eval-dir /data/sql2ast/eval_results \
  --train-data /data/sql2ast/train/train.json \
  --database-root /data/sql2ast/train/train_databases \
  --output /data/sql2ast/data/dpo_pairs.jsonl \
  --limit 100
```

### 9.2 输出内容

输出文件：

```text
/data/sql2ast/data/dpo_pairs.jsonl
```

每行通常包含：

- `prompt`
- `chosen`
- `rejected`
- `margin`
- `sample_id`
- `db_id`
- `pair_type`
- `chosen_is_correct`
- `rejected_is_correct`
- `chosen_distance`
- `rejected_distance`

### 9.3 重要说明

当前仓库的 `build_pairs.py` 会构造两类 pair：

- `(correct, wrong)`：执行正确 SQL 作为 `chosen`，执行错误 SQL 作为 `rejected`
- `(wrong, wrong)`：两条都执行错误时，AST 距离更小的 SQL 作为 `chosen`，距离更大的 SQL 作为 `rejected`

默认会保留所有可配对 pair；如果设置 `--max-pairs` 为大于 0 的值，则按 `margin` 降序截断。`correct -> wrong` 即使 wrong 的 AST 距离更小，也仍然保持 correct 作为 `chosen`，此时 `margin` 记为 0。

新版本还会把 pair 元数据一起写入 JSONL，便于后续清洗与分层训练：

- `pair_type`: `correct_wrong` 或 `wrong_wrong`
- `chosen_is_correct` / `rejected_is_correct`: 执行是否正确
- `chosen_distance` / `rejected_distance`: 相对 gold SQL 的 AST 距离

### 9.4 当前推荐的 DPO 过滤命令（CHES）

在 fully indexed 的全量 pair 文件已经准备好的前提下，当前推荐的过滤命令是：

```bash
python src/filter_dpo_pairs.py \
  --input /workspace/data/ches/data/20260428_refresh/dpo_pairs.with_meta.full.cscsql_prompt.jsonl \
  --output /workspace/data/ches/data/20260428_refresh/dpo_pairs.strict_mix.cscsql_prompt.2048_2560.jsonl \
  --summary-json /workspace/data/ches/data/20260428_refresh/dpo_pairs.strict_mix.cscsql_prompt.2048_2560.summary.json \
  --tokenizer /workspace/models/Qwen3-4B-Instruct-2507 \
  --allowed-pair-types correct_wrong,wrong_wrong \
  --min-margin 0.05 \
  --max-prompt-tokens 2048 \
  --max-total-tokens 2560 \
  --max-pairs-per-prompt 4
```

## 10. 阶段五：SFT 训练

### 10.1 修改配置

先编辑 [`configs/sft.yaml`](configs/sft.yaml)，至少确认这几个字段：

```yaml
model_name_or_path: "Qwen/Qwen2.5-Coder-7B-Instruct"
train_data_path: "/data/sql2ast/train/train.json"
database_root: "/data/sql2ast/train/train_databases"
output_dir: "/data/sql2ast/outputs/sft"
```

如果你当前跑的是 CHES 推荐流程，`train_data_path` 建议直接改成：

```yaml
train_data_path: "/workspace/data/ches/data/20260428_refresh/sft_augmented.recommended.json"
database_root: "/workspace/data/ches/train_databases"
```

另外，当前 `src/train_sft.py` 已经不再使用旧的 `Table xxx(...)` 简化 schema prompt，而是通过 `src/utils/cscsql_prompt.py` 复用 `csc_sql` 的 `CREATE TABLE ...` prompt 逻辑。  
只要 `train_db_contents_index` 已准备好，训练时会自动把 question-related DB values 融入 prompt。

### 10.2 单机 8 卡启动

如果你希望 SFT / DPO 自动上报到 Weights & Biases，先复制环境模板并填写密钥：

```bash
cp configs/wandb.local.env.example configs/wandb.local.env
```

`scripts/02_train_sft.sh` 和 `scripts/03_train_dpo.sh` 会在启动时自动 `source configs/wandb.local.env`。如果你是直接手写 `torchrun` 命令调用 `src/train_sft.py` / `src/train_dpo.py`，则需要先手动 `source configs/wandb.local.env`，并把 `report_to` 设为 `"wandb"`。

推荐在 Linux 服务器上使用 `torchrun`：

```bash
torchrun --nproc_per_node=8 src/train_sft.py --config configs/sft.yaml
```

如果只想先试单卡冒烟：

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py --config configs/sft.yaml
```

### 10.3 默认配置含义

默认 `configs/sft.yaml` 中：

- `per_device_train_batch_size: 4`
- `gradient_accumulation_steps: 4`

单卡有效 batch size 为：

```text
4 x 4 = 16
```

8 卡总有效 batch size 为：

```text
4 x 4 x 8 = 128
```

对 8 x A100 40G 来说，Qwen2.5-Coder-7B + LoRA + bf16 通常是合理起点。如果显存仍然紧张，可以优先降低：

- `per_device_train_batch_size`
- `max_seq_length`

### 10.4 训练产物

训练完成后，默认输出到：

```text
/data/sql2ast/outputs/sft
```

该目录会作为下一阶段 DPO 的初始模型。

## 11. 阶段六：DPO 训练

### 11.1 修改配置

编辑 [`configs/dpo.yaml`](configs/dpo.yaml)，至少确认：

```yaml
model_name_or_path: "/data/sql2ast/outputs/sft"
ref_model_name_or_path: null
dpo_pairs_path: "/data/sql2ast/data/dpo_pairs.jsonl"
output_dir: "/data/sql2ast/outputs/dpo"
```

如果你当前跑的是 CHES 推荐流程，建议将 `dpo_pairs_path` 指向：

```yaml
dpo_pairs_path: "/workspace/data/ches/data/20260428_refresh/dpo_pairs.strict_mix.cscsql_prompt.2048_2560.jsonl"
```

如果你是从 base 模型直接做 DPO，而不是从 `outputs/sft` 继续训练，则把：

```yaml
model_name_or_path: "/workspace/models/Qwen3-4B-Instruct-2507"
```

并保留：

```yaml
ref_model_name_or_path: null
```

### 11.2 单机 8 卡启动

```bash
torchrun --nproc_per_node=8 src/train_dpo.py --config configs/dpo.yaml
```

### 11.3 默认配置含义

默认 `configs/dpo.yaml` 中：

- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 8`

单卡有效 batch size 为：

```text
2 x 8 = 16
```

8 卡总有效 batch size 为：

```text
2 x 8 x 8 = 128
```

其中关键超参数：

- `beta`
  DPO 中参考模型约束强度
- `alpha`
  额外加入 AST 距离 `margin` 的权重

当前实现的 DPO loss 形式为：

```text
L = -log(sigmoid(beta * (policy_logratio - ref_logratio) + alpha * margin))
```

也就是说：

- SFT 阶段不使用 AST 距离
- DPO 阶段才会使用 `margin`

### 11.4 训练产物

训练完成后，默认输出到：

```text
/data/sql2ast/outputs/dpo
```

## 12. 推荐的完整命令顺序

下面是一套适合 Linux 云服务器的完整执行顺序。请先按你的实际路径修改。

```bash
# 1) 安装依赖
pip install -r requirements.txt
pip install pyyaml datasets transformers peft trl accelerate sentencepiece
pip install scipy scikit-learn

# 2) 执行候选 SQL 评估
python eval.py --output-dir /data/sql2ast/eval_results --num-cpus 8

# 3) 导出 AST / 规范化 SQL
python sql_to_ast.py /data/sql2ast/eval_results \
  -o /data/sql2ast/eval_results_ast \
  --pattern '*_eval.json' \
  --dialect sqlite

# 4) 校准距离函数
python src/calibrate.py \
  --eval-dir /data/sql2ast/eval_results \
  --train-data /data/sql2ast/train/train.json \
  --database-root /data/sql2ast/train/train_databases \
  --output /data/sql2ast/reports/distance_calibration.json

# 5) 构造 DPO 数据
python src/build_pairs.py \
  --eval-dir /data/sql2ast/eval_results \
  --train-data /data/sql2ast/train/train.json \
  --database-root /data/sql2ast/train/train_databases \
  --output /data/sql2ast/data/dpo_pairs.jsonl

# 6) 先做 SFT
torchrun --nproc_per_node=8 src/train_sft.py --config configs/sft.yaml

# 7) 再做 DPO
torchrun --nproc_per_node=8 src/train_dpo.py --config configs/dpo.yaml
```

## 13. 建议的实验节奏

为了避免在大机器上长时间跑错配置，建议按下面节奏推进：

1. 先用 `--limit 5` 跑 `eval.py`
2. 检查 `summary.json` 和几个 `*_eval.json`
3. 再跑 `sql_to_ast.py`
4. 抽样检查 AST 转换结果和 parse error
5. 先用少量样本跑 `build_pairs.py`
6. 抽样检查 `dpo_pairs.jsonl`
7. 单卡跑 SFT / DPO 冒烟
8. 最后切到 8 卡正式训练

这样可以显著减少大规模训练前的无效消耗。

## 14. 常见问题

### 14.1 为什么 `requirements.txt` 很短？

因为它当前只覆盖 AST 相关基础依赖，没有覆盖训练依赖。训练前请按本文档额外安装：

- `pyyaml`
- `datasets`
- `transformers`
- `peft`
- `trl`
- `accelerate`
- `scipy`
- `scikit-learn`
- `torch`

### 14.2 为什么 SFT 要先于 DPO？

因为 DPO 假设模型已经具备基本生成能力。这个仓库的默认配置也是让 `src/train_dpo.py` 从 `outputs/sft` 继续训练。

### 14.3 AST 距离在哪个阶段生效？

当前代码中：

- SFT 不使用 AST 距离
- DPO 数据构造和 DPO loss 会使用 AST 距离

### 14.4 是否可以直接开始正式 DPO？

不建议。请先人工检查 `dpo_pairs.jsonl` 的质量，确认正负样本方向符合你的训练目标。

## 15. 当前仓库的已知注意点

在正式大规模实验前，建议你额外注意下面几点：

- 当前测试基线不是全绿，建议先执行 `pytest -q`
- `src/build_pairs.py` 当前的 pair 构造策略更适合作为实验起点，不建议未经抽样验证就直接用于正式训练
- 训练脚本默认使用 LoRA 和 bf16，比较适合 A100 40G，但具体 batch size 仍建议先单卡冒烟确认

## 16. 最终闭环

这套仓库的完整链路可以概括为：

```text
候选 SQL -> 执行评估 -> AST/距离分析 -> 构造偏好数据 -> SFT -> DPO
```

如果你后续还要做正式 benchmark，建议再补充一套“模型生成测试集 SQL -> 执行评估”的推理脚本，把训练闭环和测试闭环完全接上。
