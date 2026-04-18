# SQL2AST 训练方案:基于分层复合 AST 距离的 Margin-aware DPO

## 1. 项目目标

用 **AST 分层复合距离** 作为稠密奖励信号,训练一个 7B 级开源 coder 模型,使其在 Spider / BIRD 基准上的 Execution Accuracy(EX)接近甚至达到 GPT-4 级别。

核心链路:
```
NL + Schema → Base LLM 采样 K 条候选 SQL
             → 对每条候选用分层距离 D(candidate, gold) 打分
             → 构造带 margin 的偏好对
             → Margin-aware DPO 微调
             → 得到 M_dpo
```

## 2. 技术栈与依赖

| 组件 | 选型 | 说明 |
|---|---|---|
| Base 模型 | Qwen2.5-Coder-7B-Instruct | LearNAT 同款,开源,上下文 32K |
| SQL 解析 | sqlglot | 支持多方言、有 diff、有 optimizer |
| 训练框架 | TRL(DPOTrainer)+ PEFT + Accelerate | 轻量、可 LoRA |
| 数据集 | Spider(train/dev)+ BIRD(train/dev) | 主流 NL2SQL 基准 |
| 执行验证 | SQLite3 | Spider/BIRD 原生数据库 |
| 硬件 | 单卡 A100 80G 或 2×RTX 4090 | LoRA 训练足够 |

依赖清单:
```
sqlglot>=23.0
transformers>=4.40
trl>=0.9
peft>=0.10
accelerate>=0.30
datasets
deepspeed   # 可选
bitsandbytes # 4bit 量化可选
```

## 3. 分阶段执行计划

总时长估算:**15~20 个工作日**(单人单卡),可并行化进一步压缩。

### Phase 0:环境与基线(1-2 天)

**任务**
1. 安装依赖,下载 Spider/BIRD 数据集 + databases
2. 加载 Qwen2.5-Coder-7B-Instruct,跑通一个纯 prompt(zero-shot)推理 baseline
3. 在 Spider dev 上记录 baseline EX,作为对照

**产出物**
- `scripts/env_setup.sh`
- `reports/baseline_zero_shot.md`(EX/EM 数字)

---

### Phase 1:规范化 + 分层距离函数(3-5 天,关键)

这是整个项目的**地基**。距离函数不靠谱,后面全白干。

**1.1 SQL 规范化管线 `src/normalize.py`**

顺序执行:
1. `sqlglot.parse_one(sql, dialect="sqlite")`(BIRD 是 SQLite,Spider 混合)
2. `qualify_tables`:展开表别名
3. `qualify_columns(schema)`:给裸列加表前缀、展开 `SELECT *`
4. 归一化 JOIN 类型:把不带 kind 的 JOIN 统一成 INNER JOIN
5. 归一化字面量:`'1' → 1`(数字列)、`TRUE → 1`、`FALSE → 0`
6. 对无序子句排序:`WHERE` AND 链、`SELECT` 列按字面量排序
7. `sqlglot.optimizer.simplify`:消除恒真/恒假表达式

**1.2 三个子距离 `src/distance/`**

**(a) Component-F1(主干,权重 0.5)**

把 SQL 拆成 7 个组件集合,每个算 F1,最后平均:

| 组件 | 元素表示 |
|---|---|
| Tables | `{table_name, ...}` |
| Join graph | `{frozenset({table_a, table_b}), ...}` |
| Select items | `{(agg_fn, column), ...}` |
| Where predicates | `{(column, op, normalized_value), ...}` |
| Group by | `{column, ...}` |
| Order by | `{(column, direction), ...}` |
| Having + Limit | 独立小集合 |

F1(集合 A, 集合 B) = `2·|A∩B| / (|A|+|B|)`,空集对空集记为 1.0。

**(b) TED(结构信号,权重 0.3)**

```python
from sqlglot.diff import diff, Keep
edits = diff(ast1, ast2)
ted = sum(1 for e in edits if not isinstance(e, Keep))
sim_struct = 1 - ted / max(count_nodes(ast1), count_nodes(ast2))
```

**(c) Schema penalty(罚项,权重 0.2)**

专门惩罚 schema linking 错误(表/列名写错是 NL2SQL 最常见的低级错误):
- 对 sampled 用到但 gold 没用的每张表/每个列,`+0.1` 罚分
- clip 到 [0, 1]

**1.3 分层复合距离**

```python
def hierarchical_distance(sampled_sql, gold_sql, schema):
    # 规范化
    s = normalize(sampled_sql, schema)
    g = normalize(gold_sql, schema)
    
    # 不可解析惩罚
    if s is None:
        return 1.0  # 最大距离,最强负样本
    
    d_comp = 1 - component_f1(s, g)
    d_ted  = ted_normalized(s, g)
    d_sch  = schema_penalty(s, g, schema)
    
    return 0.5 * d_comp + 0.3 * d_ted + 0.2 * d_sch  # ∈ [0, 1]
```

**1.4 距离函数校准(不可跳过)**

在 Spider dev 上:
1. 对每条样本,用 base 模型采样 8 条候选
2. 分别算 (a) 我们的 D 距离 (b) 执行是否正确 EX
3. 计算 `Spearman(D, 1-EX)` 相关系数
4. **目标:相关系数 ≥ 0.65**。低于这个数说明距离函数的权重或组件划分有问题,回去调

**产出物**
- `src/normalize.py`、`src/distance/*.py`
- `tests/test_distance.py`(至少 20 个单元测试覆盖等价重写、列名错、结构错等)
- `reports/distance_calibration.md`(相关性分析)

---

### Phase 2:SFT 热身(2-3 天)

**为什么要 SFT**:直接对 base 模型做 DPO,采样候选质量太差,偏好对信号近乎随机。SFT 先把模型抬到"能写出可解析 SQL"的水平。

**配置**
- 数据:Spider train + BIRD train,共约 10K 条
- Prompt 格式:`<schema>\n<question>\n\nSQL:`
- LoRA: r=16, alpha=32, dropout=0.05, 目标模块 q/k/v/o_proj
- 超参:lr=2e-4, batch=16, epoch=2, warmup 0.03
- 序列长度 2048

**验收标准**
- Spider dev 上 EX 提升 15% 以上(vs zero-shot baseline)
- 采样候选中**可解析率 ≥ 85%**(否则 DPO 阶段会采到太多垃圾)

**产出物**:`checkpoints/M_sft/`

---

### Phase 3:采样与偏好对构造(2-3 天)

**3.1 候选采样 `src/sample.py`**

对 Spider+BIRD train 的每条样本,用 M_sft:
- 温度 0.8,top_p=0.95
- 采样 K=8 条候选
- 记录每条的生成 logprob(DPO 会用到)

**3.2 距离打分**

对 8K×8 = 64K 条候选分别算 D,得到列表。

**3.3 构造偏好对 `src/build_pairs.py`**

对每条样本内的 8 条候选:
1. 按 D 升序排序
2. 生成所有 `C(8,2)=28` 对,chosen 选 D 小的
3. 保留 `margin = D_l − D_w ≥ 0.05` 的对子
4. 每个样本 **最多留 6 对**(避免某些样本主导训练)
5. 去重:同一个 (chosen, rejected) 文本只留一次

**3.4 特殊情形**
- 8 条全部 D=1(都不可解析或完全错)→ 整个样本丢弃
- 8 条全部 D=0(都完美)→ 丢弃(没信号)
- 不可解析的候选永远参与构造,作为强负样本(D=1)

**预期输出规模**:约 20K~40K 条偏好对,JSONL 格式:
```json
{"prompt": "...", "chosen": "...", "rejected": "...", "margin": 0.42}
```

**产出物**
- `data/preference_pairs/{spider,bird}.jsonl`
- `reports/pair_statistics.md`(margin 分布、正负样本质量)

---

### Phase 4:Margin-aware DPO 训练(3-5 天,核心)

**4.1 自定义损失**

TRL 的 `DPOTrainer` 默认损失:
```
L = -log σ(β · (r_w - r_l))
```

扩展为 margin 版:
```
L = -log σ(β · (r_w - r_l) - α · margin)
```

实现方式(继承 DPOTrainer,覆盖 `dpo_loss`):
```python
class MarginAwareDPOTrainer(DPOTrainer):
    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps,
                 reference_chosen_logps, reference_rejected_logps, margin):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = self.beta * (pi_logratios - ref_logratios) - self.alpha * margin
        losses = -F.logsigmoid(logits)
        return losses, ...
```

需要在 collator 里把 `margin` 字段打包进 batch。

**4.2 超参**

| 参数 | 值 | 说明 |
|---|---|---|
| β | 0.1 | DPO 温度,NL2SQL 比通用 chat 小 |
| α | 1.0 | margin 缩放,先用 1,后续实验 0.5/2 |
| lr | 5e-7 | 比 SFT 小 2 个数量级 |
| batch | 32(gradient acc) | 单卡放不下大 batch |
| epoch | 1 | 警惕过拟合 |
| warmup | 0.1 | |
| LoRA | 继承 M_sft 的 LoRA,不 merge | |

**4.3 训练监控指标**

每 50 步打印:
- `train/loss`
- `train/chosen_rewards`、`train/rejected_rewards`
- `train/accuracy`(chosen logprob > rejected logprob 的比例,**目标 ≥ 0.7**)
- `train/margins`(policy margin 均值,应持续上升)
- `train/kl_to_ref`(应保持在合理范围,比如 < 10,太大说明训飞了)

**4.4 早停**
- 每 500 步在 Spider dev 100 条子集上跑 EX
- 连续 2 次不涨就早停

**产出物**:`checkpoints/M_dpo/`

---

### Phase 5:评估与错误分析(2-3 天)

**5.1 主评估**

| 数据集 | 指标 | 目标 |
|---|---|---|
| Spider dev | EX | 超过 M_sft ≥ 5% |
| Spider test | EX | 超过 M_sft ≥ 3% |
| BIRD dev | EX | 超过 M_sft ≥ 5% |
| BIRD dev | VES(Valid Efficiency) | 不能下降 |

**5.2 错误分解**

对 M_dpo 答错的样本,按距离**组件归因**:
- 仅 Component-F1 失败 → 逻辑错(谓词/聚合错)
- 仅 Schema penalty 失败 → schema linking 错
- 仅 TED 失败 → 结构冗余或缺失子句
- 混合失败 → 严重错误

出一张饼图,指导下一轮改进方向。

**5.3 消融实验**

| 实验 | 目的 |
|---|---|
| 去掉 TED(w_ted=0) | 验证结构信号必要性 |
| 去掉 Schema penalty | 验证 schema linking 惩罚 |
| α=0(退化为标准 DPO) | 验证 margin 的价值 |
| K=4 vs K=8 vs K=16 | 验证采样预算影响 |

**产出物**
- `reports/final_eval.md`
- `reports/ablation.md`
- `reports/error_analysis.md`

---

## 4. 项目文件结构

```
SQL2AST/
├── data/
│   ├── spider/                  # 原始数据集
│   ├── bird/
│   └── preference_pairs/        # Phase 3 产出
│       ├── spider.jsonl
│       └── bird.jsonl
├── src/
│   ├── normalize.py             # sqlglot 规范化
│   ├── distance/
│   │   ├── component_f1.py
│   │   ├── ted.py
│   │   ├── schema_penalty.py
│   │   └── composite.py         # 分层距离入口
│   ├── sample.py                # 候选采样
│   ├── build_pairs.py           # 偏好对构造
│   ├── train_sft.py
│   ├── train_dpo.py             # 含 MarginAwareDPOTrainer
│   ├── eval.py                  # EX / EM / 分布诊断
│   └── utils/
│       ├── schema.py            # schema 加载与 qualify
│       └── executor.py          # SQLite 执行与结果对比
├── tests/
│   ├── test_normalize.py
│   ├── test_distance.py
│   └── fixtures/                # 20+ 等价/非等价 SQL 对
├── configs/
│   ├── sft.yaml
│   └── dpo.yaml
├── scripts/
│   ├── 00_env_setup.sh
│   ├── 01_baseline.sh
│   ├── 02_calibrate_distance.sh
│   ├── 03_train_sft.sh
│   ├── 04_sample_candidates.sh
│   ├── 05_build_pairs.sh
│   ├── 06_train_dpo.sh
│   └── 07_evaluate.sh
├── checkpoints/
│   ├── M_sft/
│   └── M_dpo/
├── reports/                     # 每个阶段的实验报告
├── TRAINING_PLAN.md             # 本文件
└── README.md
```

## 5. 风险与应对

| 风险 | 现象 | 应对 |
|---|---|---|
| 距离与 EX 相关性低 | Phase 1.4 校准 Spearman < 0.5 | 调整组件权重;Where predicates 值归一化加强;考虑引入轻量执行反馈 |
| SFT 后可解析率低 | < 85% | 增加 SFT epoch,或检查 prompt 格式 |
| DPO 训飞 | KL 快速上升,EX 反降 | 降 β 到 0.05,降 lr,减 epoch |
| 偏好对质量差 | 大量 chosen 自己也错 | 收紧 margin 阈值到 0.1;只保留"chosen 的 D ≤ 0.3"的对子 |
| 不可解析候选占比高 | Phase 3 超过 20% | 回 SFT 补训,或 base 模型换更大的 14B |
| 过拟合偏好对 | train acc 飙到 0.95,test EX 不涨 | 早停,或用更多样本 |
| Spider/BIRD schema 差异大 | 模型在 BIRD 上效果差 | 分开训两个 LoRA 适配器 |

## 6. 里程碑与验收

| 里程碑 | 时间点 | 验收条件 |
|---|---|---|
| M1:距离函数可用 | Day 5 | Spider dev 上 Spearman ≥ 0.65 |
| M2:SFT 基线就位 | Day 8 | Spider dev EX ≥ zero-shot + 15%,可解析率 ≥ 85% |
| M3:偏好对数据就绪 | Day 11 | ≥ 20K 对,margin 分布合理 |
| M4:DPO 模型 | Day 16 | Spider dev EX ≥ M_sft + 5% |
| M5:评估与报告 | Day 20 | Spider/BIRD 全指标 + 消融 + 错误分析完成 |

## 7. 后续拓展(P2,视 M4 结果决定)

- **自举循环**:用 M_dpo 重新采样 → 新偏好对 → 再训 M_dpo_v2
- **步级 DPO**:结合 LearNAT 的 MCTS 分解,做 subtask 粒度的偏好学习
- **推理时 medoid 重排**:训练无关,立刻加分
- **多方言适配**:把 Postgres/MySQL 方言纳入 sqlglot 规范化

---

**关键原则**

1. **距离函数优先**:Phase 1 必须扎实,是全项目的 load-bearing 组件
2. **小步快跑**:每个 Phase 结束都要出一份 report,不要一路跑到最后才回头看
3. **消融先行**:在 M4 之前把 α=0 的对照组跑了,确认 margin 真的有效
4. **EX 是唯一金标准**:AST 距离只是训练信号,评估看 EX
