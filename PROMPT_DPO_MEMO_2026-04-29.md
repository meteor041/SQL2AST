# Prompt / DPO Memo 2026-04-29

## 1. Prompt 对齐状态

- 已把 `sql_rm` 的 SFT / DPO prompt 构造切到 `csc_sql` 路径。
- 代码入口：
  - `src/utils/cscsql_prompt.py`
  - `src/train_sft.py`
  - `src/build_pairs.py`
- 现在不再使用旧的 `Table xxx(...)` 简化 schema prompt。
- 当前 prompt 主骨架已经和 `csc_sql` 对齐：
  - `Task Overview`
  - `Database Engine`
  - `CREATE TABLE ...`
  - `Question`
  - `Instructions`
  - `SQL:`

## 2. Java / csc_sql 运行状态

- 已安装 Java:
  - `java 21.0.7`
  - `javac 21.0.7`
- `cscsql.service.process.process_dataset` 已可正常 import。

## 3. relevant hits 支持状态

- 已在 `src/utils/cscsql_prompt.py` 中补上 relevant hits 支持。
- 行为：
  - 如果存在 `db_contents_index`，自动使用 `LuceneSearcher` 检索 question-related DB values。
  - 如果不存在索引，自动退回 sampled-values-only，不报错。
- 已新增索引构建脚本：
  - `src/build_cscsql_contents_index.py`
- 这个脚本支持当前 CHES 的 flat / nested sqlite 布局。

## 4. 当前 relevant hits 实际落地情况

- 训练库索引目录：
  - `/workspace/data/ches/train_db_contents_index`
- 已验证前几个库可以成功建索引并被 prompt 适配层读取。
- 今天构建过程中确认至少前 4 个库已落地。
- 这意味着：
  - 对这些已建索引的库，prompt 已经能看到更多 `-- example:`。
  - 对尚未建索引的库，仍然是 sampled-values-only。

## 5. Prompt 抽样观察

- 旧问题确认：
  - 有些 prompt 只有 `CREATE TABLE`，没有多少 `-- example:`
  - 原因不是模板没切，而是 relevant hits / sampled values 信息不够
- 新验证：
  - `address` 库在 relevant hits 启用后，`example_count = 72`
  - 说明 richer prompt 这条路是通的

## 6. DPO 文件现状

已新生成的文件：

- 全量 with-meta:
  - `/workspace/data/ches/data/20260428_refresh/dpo_pairs.with_meta.full.cscsql_prompt.jsonl`
- strict, 旧长度阈值 `768/1024`:
  - `/workspace/data/ches/data/20260428_refresh/dpo_pairs.full.strict.cscsql_prompt.jsonl`
- strict, 当前更可用阈值 `1024/1536`:
  - `/workspace/data/ches/data/20260428_refresh/dpo_pairs.full.strict.cscsql_prompt.1024_1536.jsonl`

注意：

- 上述 DPO 文件是在 prompt 切到 `csc_sql` 之后重写的。
- 但它们是在“训练库索引未全量建完”之前生成的。
- 所以这些 DPO 文件还不是 fully relevant-hits-enabled 版本。

## 7. DPO 数量统计

strict 候选池，长度过滤前：

- `52,513` 对

最终 strict 数据量：

- `768/1024`: `6,867`
- `1024/1536`: `9,903`
- `2048/2560`: `12,986`
- `2560/3072`: `13,398`

## 8. max_len 结论

- `1024/1536 -> 2048/2560` 增益明显。
- `2048/2560 -> 2560/3072` 增益很小。
- 当前最均衡推荐档位：
  - `max_prompt_tokens = 2048`
  - `max_total_tokens = 2560`
- 真正的大瓶颈不只是长度，还有：
  - `max_pairs_per_prompt = 4`

## 9. DPO 质量抽查结论

- 抽查 10 条后，数据不是不能用，但噪声偏高。
- 主要问题：
  - `chosen` 有时并不明显优于 `rejected`
  - `execution-correct` 不等于“更符合题意”
  - 某些 `gold SQL` 本身也不是最自然答案
- 当前判断：
  - 这批 strict DPO 可以作为实验起点
  - 但不适合不加筛查就直接长训

## 10. 明天建议的优先顺序

1. 先把 `/workspace/data/ches/train_db_contents_index` 全量建完。
2. 用 fully indexed 状态重新抽样 SFT prompt，确认 richer prompt 覆盖面。
3. 重新重写一版 DPO full / strict 文件。
4. 对新 DPO 数据再做一次 20-50 条抽查。
5. 再决定：
   - 是否直接跑 DPO
   - 是否需要更保守过滤
   - 是否把 `per_prompt_cap` 从 `4` 调到 `6` 或 `8`

## 11. 2026-04-30 已完成

- `/workspace/data/ches/train_db_contents_index` 已全量建完。
- fully indexed 的 full DPO 文件已重写完成：
  - `/workspace/data/ches/data/20260428_refresh/dpo_pairs.with_meta.full.cscsql_prompt.jsonl`
- fully indexed 的 strict mix DPO 训练文件（`min_margin=0.05`）已生成完成：
  - `/workspace/data/ches/data/20260428_refresh/dpo_pairs.strict_mix.cscsql_prompt.2048_2560.jsonl`
  - `/workspace/data/ches/data/20260428_refresh/dpo_pairs.strict_mix.cscsql_prompt.2048_2560.summary.json`
- 最终统计：
  - total pairs: `18,700`
  - `correct_wrong`: `10,042`
  - `wrong_wrong`: `8,658`
  - unique prompts: `4,935`

## 12. 明天直接跑 DPO 的命令

系统时区已经改成 `Asia/Shanghai`，所以 `date` 生成的时间戳将是上海时间。

推荐明天直接使用 `min_margin=0.05` 的 fully indexed 训练文件，命令如下：

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

export SFT_OUTPUT=/workspace/data/ches/outputs/sft_20260429_135759_augmented_eval1pct_wandb_h100
export DPO_PAIRS_PATH=/workspace/data/ches/data/20260428_refresh/dpo_pairs.strict_mix.cscsql_prompt.2048_2560.jsonl

export DPO_MAX_PROMPT_LENGTH=1024
export DPO_MAX_SEQ_LENGTH=1536

export DPO_OUTPUT=/workspace/data/ches/outputs/dpo_$(date +%Y%m%d_%H%M%S)_strict_mix_cscsql_prompt_2048_2560_margin005_from_sft_20260429_135759_h100
export DPO_RUN_NAME=sql_rm-dpo-$(date +%Y%m%d_%H%M%S)-strict-mix-cscsql-2048-2560-m005

bash scripts/03_train_dpo.sh
```

备注：

- 这里先保留当前较稳的训练长度：
  - `max_prompt_length = 1024`
  - `max_seq_length = 1536`
- 训练文件过滤长度比训练长度更宽（`2048/2560`）是接受的；训练时会有截断。
- 如果明天想尝试更激进的长度，再单独评估显存与训练时长。
