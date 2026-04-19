# 代码文件说明

## 项目结构总览

```
SQL2AST/
├── eval.py                      # 执行评估：采样 SQL → correct/wrong 分类
├── sql_to_ast.py                # SQL 规范化存档工具（可选预检步骤）
│
├── src/                         # 核心训练代码
│   ├── normalize.py             # SQL 语义规范化
│   ├── utils/
│   │   ├── schema.py            # 从 SQLite 读取表结构
│   │   └── prompt.py            # 格式化 LLM prompt
│   ├── distance/
│   │   ├── component_f1.py      # 子距离①：7 组件 F1
│   │   ├── ted.py               # 子距离②：Tree Edit Distance
│   │   ├── schema_penalty.py    # 子距离③：Schema 违规罚项
│   │   └── composite.py         # 分层复合距离入口
│   ├── build_pairs.py           # 构造 DPO 偏好对 JSONL
│   ├── calibrate.py             # Spearman 相关性校准
│   ├── train_sft.py             # SFT 热身训练
│   └── train_dpo.py             # Margin-aware DPO 训练
│
├── configs/
│   ├── sft.yaml                 # SFT 超参配置
│   └── dpo.yaml                 # DPO 超参配置
│
├── tests/
│   └── test_distance.py         # 距离函数单元测试（40 个）
│
└── cscsql/utils/                # 原有工具库（不修改）
    ├── sqlite_db_utils.py       # SQLite 执行封装
    └── file_utils.py            # 文件读写工具
```

---

## 数据流

```
BIRD train.json + 16条采样SQL
        │
        ▼
eval.py
  输入：采样文件目录（每文件含 all_sqls 列表）
  输出：eval_results/*_eval.json
        每文件含：correct_set / wrong_set / metadata(sample_id, db_id)
        │
        ▼
src/build_pairs.py
  对每条 query 的所有候选算 D(candidate, gold)
  构造 (chosen, rejected, margin) 偏好对
  输出：data/dpo_pairs.jsonl
        │
        ├──▶ src/train_sft.py  →  outputs/sft/
        │
        └──▶ src/train_dpo.py  →  outputs/dpo/
```

---

## 各文件详细说明

---

### `eval.py`

**职责**：对每条 query 的多条候选 SQL，通过实际执行 SQLite 判断正确性。

**输入**：
- `.location` 配置文件，指定 `TRAIN_DATA_PATH`、`TRAIN_DATABASE_PATH`、`SQL_PATH`
- 采样 SQL 文件，命名格式 `{sample_id}_{db_id}.json`，每文件含 `all_sqls` 列表

**输出**：`eval_results/{sample_id}_{db_id}_eval.json`，格式：
```json
{
  "metadata": {"sample_id": 0, "db_id": "movies"},
  "correct_set": [{"sql": "...", "gold_sql": "...", "gold_result": {...}}],
  "wrong_set":   [{"sql": "...", "gold_sql": "...", "pred_result": {...}}]
}
```

**关键函数**：
- `load_sql_clusters(path)` — 把 `all_sqls` 列表按 SQL 文本去重聚类
- `execute_sql(sqlite_path, sql, timeout)` — 执行 SQL，返回 `QueryResult`
- `results_equal(gold, pred, ignore_order)` — 比对执行结果
- `evaluate_file(...)` — 处理单个采样文件，返回 `(correct_records, wrong_records, error)`

---

### `sql_to_ast.py`

**职责**：可选的预检/存档工具。把 eval_results 或原始采样文件里的 SQL 规范化（parse + re-serialize），存储 `normalized_sql` 字符串供人工检视。

**不再做**：不序列化 AST dict（已移除 `expression_to_dict`）。

**两种输入模式**：
1. `*_eval.json`（含 `correct_set`）→ 输出含 `gold / correct / wrong` 三组的 JSON
2. 原始采样文件（含 `all_sqls`）→ 在每条记录上追加 `normalized_sqls` 字段

**何时用**：在 `eval.py` 和 `build_pairs.py` 之间，如需人工核查解析失败率时运行；正常自动化流程可跳过。

---

### `src/utils/schema.py`

**职责**：从 SQLite 文件读取表结构，供规范化和距离计算使用。

**核心数据结构**：
```python
DBSchema
├── db_id: str
└── tables: dict[str, TableSchema]   # key 为小写表名
    └── TableSchema
        ├── name: str                # 原始大小写
        ├── columns: list[ColumnInfo]
        └── foreign_keys: list[tuple]
```

**主要函数**：

| 函数 | 说明 |
|---|---|
| `load_schema(db_path)` | 读 PRAGMA table_info + foreign_key_list，返回 DBSchema |
| `get_column_set(schema)` | 返回所有 `"table.column"` 字符串（小写），供 schema_penalty 使用 |
| `get_table_set(schema)` | 返回所有表名（小写）集合 |
| `schema_to_sqlglot_dict(schema)` | 转为 `{table: {col: type}}`，供 sqlglot qualify 使用 |
| `schema_to_prompt_dict(schema)` | 转为 `{table: [col, ...]}` ，供 prompt 格式化使用 |

---

### `src/utils/prompt.py`

**职责**：把 question + schema + evidence 格式化为统一的 LLM prompt 字符串。SFT 和 DPO 使用相同格式，确保分布一致。

**输出示例**：
```
Schema:
Table movies(movie_id, movie_title, release_year, movie_popularity)
Table ratings(rating_id, movie_id, rating_score, user_id)

Question: 1945年上映的电影按popularity降序列出名称

Evidence: released in year 1945 refers to release_year = 1945

SQL:
```

**主要函数**：

| 函数 | 说明 |
|---|---|
| `format_schema_section(schema_dict, fks)` | 把 schema dict 渲染为文本块 |
| `format_nl2sql_prompt(question, schema_dict, evidence, ...)` | 拼装完整 prompt（不含 SQL） |
| `format_sql_response(sql)` | 把 SQL strip 后作为 assistant 回复 |

---

### `src/normalize.py`

**职责**：SQL 语义规范化。核心目标是让等价 SQL 在进入距离函数之前尽量对齐，减少因别名、列名大小写等表面差异造成的假性距离。

**规范化步骤**（按顺序，每步失败则 fallback）：
1. `sqlglot.parse_one(sql, read=dialect)` — 解析
2. `qualify(ast, schema=..., qualify_tables=True, qualify_columns=True)` — 展开表别名、给裸列加表前缀
3. `simplify(ast)` — 消除恒真/恒假条件

**三个公开函数**：

| 函数 | 输入 | 输出 | 失败行为 |
|---|---|---|---|
| `qualify_sql(sql, schema, dialect)` | SQL 字符串 | `Expression` or `None` | 返回 None |
| `normalize_sql(sql, schema, dialect)` | SQL 字符串 | 规范化后的字符串 | 三级 fallback，永不报错 |
| `parse_to_ast(sql, dialect)` | SQL 字符串 | `Expression` or `None` | 返回 None |

`normalize_sql` 的 fallback 链：qualify → parse+reserialize → strip 原字符串。

---

### `src/distance/component_f1.py`

**职责**：计算 SQL 的 7 个语义组件的 F1 距离。

**7 个组件及提取方式**：

| 组件 | 提取函数 | 元素格式 |
|---|---|---|
| select | `extract_select_cols` | `"(agg,col)"` 或 `"(none,col)"` |
| where | `extract_where_conditions` | 每个叶子谓词的 SQL 字符串 |
| groupby | `extract_groupby_cols` | 列名字符串 |
| orderby | `extract_orderby_cols` | `"(col,asc)"` 或 `"(col,desc)"` |
| having | `extract_having_conditions` | 谓词字符串 |
| limit | `extract_limit` | 数字字符串 |
| join | `extract_join_tables` | 表名字符串 |

**F1 计算**：`token_f1(pred, gold)` 使用多重集合匹配，两者均空则返回 1.0。

**最终距离**：
```
d_comp = 1 - Σ(weight_i * F1_i) / Σ(weight_i)
```
默认权重：select=2, where=2, join=2, groupby=1, orderby=1, having=1, limit=0.5。

---

### `src/distance/ted.py`

**职责**：计算两棵 AST 之间的归一化树编辑距离。

**实现**：直接调用 `sqlglot.diff(ast1, ast2)`（GumTree 算法），统计非 `Keep` 操作数。

```python
ted = edit_count / max(count_nodes(ast1), count_nodes(ast2))
```

**公开函数**：

| 函数 | 说明 |
|---|---|
| `count_nodes(ast)` | 遍历 AST 计算节点总数 |
| `normalized_ted(ast_pred, ast_gold)` | 返回 [0,1]，任一为 None 返回 1.0 |

---

### `src/distance/schema_penalty.py`

**职责**：惩罚预测 SQL 中引用了数据库中不存在的表或列。捕获 NL2SQL 最常见的低级错误（表名/列名写错）。

**罚分规则**：
- 每个非法表：`+0.2`
- 每个非法列（带表前缀且不在 schema 中）：`+0.1`
- 最终 clip 到 `[0, 1]`

**注意**：只惩罚有明确表前缀的列引用（`movies.title`），不惩罚裸列名（`title`），因为规范化前裸列无法确认归属。

---

### `src/distance/composite.py`

**职责**：三个子距离的加权组合，是整个距离系统的唯一对外入口。

**公式**：
```
D = 0.5 * d_comp + 0.3 * d_ted + 0.2 * d_sch
```

**两个公开函数**：

| 函数 | 返回 | 用途 |
|---|---|---|
| `hierarchical_distance(sql_s, sql_g, schema, dialect, weights)` | `float` | 训练/评估时调用 |
| `hierarchical_distance_with_detail(...)` | `DistanceDetail` | 调试、校准时查看分项 |

`DistanceDetail` 包含 `component_f1 / ted / schema_penalty / composite / parse_failed` 五个字段。

任一 SQL 无法解析时直接返回 `D=1.0`（最强负样本信号），不报错。

---

### `src/build_pairs.py`

**职责**：读取 `eval_results/` 下所有 `*_eval.json`，为每条 query 构造 DPO 偏好对并写入 JSONL。

**配对算法**：
1. 对 correct_set + wrong_set 中的每条候选 SQL，计算 `D(candidate, gold)`
2. `correct -> wrong`：每条执行正确 SQL 都和每条执行错误 SQL 配对
3. `wrong -> wrong`：执行错误 SQL 之间按距离排序，距离更小者作为 chosen，距离更大者作为 rejected
4. 跳过 `correct -> correct`，因为执行结果无法给出偏好方向
5. `margin = max(D_rejected - D_chosen, 0)`；`wrong -> wrong` 距离相等时跳过
6. 默认保留所有可配对 pair；如设置 `--max-pairs > 0`，按 margin 降序截断

**输出 JSONL 格式**：
```json
{"prompt": "...", "chosen": "...", "rejected": "...", "margin": 0.43, "sample_id": 0, "db_id": "movies"}
```

**CLI 用法**：
```bash
python src/build_pairs.py \
    --eval-dir eval_results \
    --train-data /data/bird/train/train.json \
    --database-root /data/bird/train/train_databases \
    --output data/dpo_pairs.jsonl
```

---

### `src/calibrate.py`

**职责**：验证距离函数是否有效——计算 D 与执行正确性的 Spearman 相关系数。

**验收标准**：`Spearman(D, EX) ≤ -0.65`（负相关，D 越大 EX 越低）。

**输出报告**：
```json
{
  "spearman_rho": -0.71,
  "n_samples": 12480,
  "mean_dist_correct": 0.18,
  "mean_dist_wrong": 0.67,
  "auc_roc": 0.83,
  "pass": true
}
```

**门禁作用**：校准不通过（rho > -0.65）时退出码为 1，提示调整距离权重，不应继续构造偏好对。

---

### `src/train_sft.py`

**职责**：SFT 热身训练，让 base 模型学会生成合法 SQL，为 DPO 阶段提供质量足够高的候选。

**数据构造**：
- 输入：BIRD `train.json`（question + gold SQL + db_id + evidence）
- 对每条样本，加载对应数据库的 schema，格式化 prompt
- 只对 SQL 部分（prompt 之后）计算 loss（prompt token 的 label 设为 -100）

**训练框架**：TRL `SFTTrainer` + PEFT `LoraConfig`。

**验收标准**（完成后手动验证）：
- Spider dev EX ≥ zero-shot baseline + 15%
- 候选 SQL 可解析率 ≥ 85%

**CLI 用法**：
```bash
python src/train_sft.py --config configs/sft.yaml
python src/train_sft.py --config configs/sft.yaml --learning_rate 1e-4  # 覆盖单个参数
```

---

### `src/train_dpo.py`

**职责**：Margin-aware DPO 训练，核心目标模型。

**损失函数**：
```
L = -log σ( β*(r_w − r_l) + α*margin )
```
- `r_w = β * log(π(chosen)/π_ref(chosen))`：chosen 的 log ratio
- `r_l`：rejected 的 log ratio
- `+α*margin`：margin 越大（chosen 和 rejected 质量差距越明显），logit 越大，loss 越小，梯度信号越强

**核心类 `MarginAwareDPOTrainer`**（继承 TRL `DPOTrainer`）：
- `compute_loss`：从 batch 中弹出 `margin` tensor，存入 `self._current_margin`
- `dpo_loss`：在标准 DPO logits 基础上加 `α * margin`

**Reference model**：默认使用 M_sft 的冻结副本（`ref_model_name_or_path=null`），与 policy 初始相同——这是标准 DPO 设置。

**监控指标**（每 `logging_steps` 步）：
- `train/accuracy`：chosen logprob > rejected logprob 的比例，目标 ≥ 0.70
- `train/chosen_rewards` / `train/rejected_rewards`：应持续分开
- KL 散度（通过 chosen/rejected rewards 间接观察）：过大说明训飞

**CLI 用法**：
```bash
python src/train_dpo.py --config configs/dpo.yaml
python src/train_dpo.py --config configs/dpo.yaml --beta 0.05 --alpha 2.0
```

---

### `configs/sft.yaml` / `configs/dpo.yaml`

所有键名与对应 dataclass（`SFTConfig` / `DPOTrainConfig`）字段一一对应，YAML 加载后直接 `**dict` 解包。

未在 YAML 中出现的键使用 dataclass 默认值；命令行 `--key value` 优先级最高。

**需要在运行前填写的必填路径**：
- `train_data_path`：BIRD train.json 路径
- `database_root`：SQLite 数据库根目录
- `dpo_pairs_path`：`build_pairs.py` 的输出路径

---

### `tests/test_distance.py`

**职责**：距离函数全链路单元测试，不依赖真实 SQLite 文件。

**fixture**：`_make_schema()` 在内存中构造一个最小 `DBSchema`（movies + ratings 两张表），所有测试共用。

**测试分组**：

| 分组 | 数量 | 覆盖内容 |
|---|---|---|
| `token_f1` | 6 | 完全相同、完全不同、部分重叠、空集、多重集 |
| 组件提取 | 9 | SELECT/WHERE/GROUP BY/ORDER BY/JOIN/LIMIT 各组件的提取正确性 |
| `component_f1_distance` | 5 | 相同 SQL=0、bounds、不同 SELECT、自定义权重 |
| `normalized_ted` | 5 | 相同=0、None 输入=1、bounds、节点计数 |
| `schema_penalty` | 5 | 表提取、列提取、合法 SQL=0、非法表>0、clip≤1 |
| `hierarchical_distance` | 8 | 相同=0、不可解析=1、bounds、排序、自定义权重、无 schema、detail |
| normalize helpers | 4 | 返回字符串、坏输入不报错、解析成功/失败 |

**运行**：
```bash
pytest tests/test_distance.py -v
```
