# Spec · Agentic RAG（001）

> 本文件只写**行为与验收**，不出现任何实现名词。任何一条被改动，都要在 `decisions.md` 留记录。
> 版本：v1 · 状态：待确认

## 1. 目的与非目标

**要解决的问题**：给一个「知道什么时候该检索、该拆几步、什么时候该停、检索不到就说不知道」的问答服务，并且它的每一点改进都能被一组固定数字证明。

**非目标**（明确不做，避免范围蔓延）：
- 不做多轮对话、不做会话记忆。一次提问 = 一次独立运行。
- 不做联网兜底。语料覆盖不到就拒答（见 ADR-003）。
- 不做 replan（不重拆子问题）。拆错了就在这一轮内走完，缺陷记进已知限制。
- 不做 Web 界面、不做鉴权、不做并发压测。

## 2. 语料

**14 篇 RAG 方向顶会论文的 arXiv 定版全文**，英文，只读、不加工（抽取规则见 ADR-014）。
全文不提交仓库，`scripts/fetch_corpus.py` 按 `data/corpus.lock.json` 里的编号与版本抓取，sha256 校验。

| 文件 | 论文 | 发表处 | 在这套语料里承担什么 |
|---|---|---|---|
| `rag-lewis` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | NeurIPS 2020 | 定义了 RAG 的两个变体（同一模型 / 分离式），是几乎所有后续论文的引用根 |
| `beir` | BEIR: A Heterogenous Benchmark for Zero-shot IR | NeurIPS 2021 D&B | 18 个数据集的构成与 BM25 强基线结论 |
| `react` | ReAct: Synergizing Reasoning and Acting | ICLR 2023 | 「思考—行动」交替，agent 循环的形态来源 |
| `ircot` | Interleaving Retrieval with Chain-of-Thought | ACL 2023 | 多跳检索的交错策略与 recall 口径 |
| `hyde` | Precise Zero-Shot Dense Retrieval (HyDE) | ACL 2023 | 用生成文档做查询扩展，与「查询改写」同源 |
| `rankgpt` | Is ChatGPT Good at Search? (RankGPT) | EMNLP 2023 | 重排：滑窗 + 排列生成，重排增益的原始证据 |
| `flare` | Active Retrieval Augmented Generation | EMNLP 2023 | 主动检索：何时触发检索的判据 |
| `lost-in-the-middle` | Lost in the Middle | TACL 2024 | 长上下文的位置效应，top-k 与顺序的证据 |
| `ragas` | Ragas: Automated Evaluation of RAG | EACL 2024 Demo | faithfulness / answer relevance / context relevance 三个指标的定义 |
| `self-rag` | Self-RAG: Learning to Retrieve, Generate, and Critique | ICLR 2024 | 反思 token 与自适应检索，多跳题的主要素材 |
| `ares` | ARES: An Automated Evaluation Framework for RAG | NAACL 2024 | 对 RAGAS 类判官方法的批评 + PPI 少量标注 |
| `adaptive-rag` | Adaptive-RAG: Adapt through Question Complexity | NAACL 2024 | 按问题复杂度分流的分类器，本项目的 route 节点的对照 |
| `crag` | CRAG — Comprehensive RAG Benchmark | NeurIPS 2024 D&B | 4,409 题、5 域 8 类，无答案/动态事实的权威口径 |
| `ragchecker` | RAGChecker: A Fine-grained Framework | NeurIPS 2024 | 检索与生成的细粒度归因指标 |

**为什么用真实论文而不是自造文档**：标签来自他人文本，不是我自证；论文互引构成天然多跳链；三篇论文都报 `recall@k` 但数值不同，天然构成近邻干扰。（ADR-014）

**三类题的取材**（问题与参考答案均为英文，全部从原文摘取并 grep 复核）：
- 单跳：某篇论文报的某个数字（CRAG 的题目总数与领域数、BEIR 的数据集数、Self-RAG 的反思 token 类别数）、某个方法的判据（FLARE 用什么信号触发检索）、某个指标的定义（RAGAS 的 faithfulness 怎么算）。
- 多跳：必须跨两个不同 `source` 文件才计入。形态是「A 论文说自己用了 X 基准 / Y 指标 → X 或 Y 的原始定义与规模写在另一篇」。
- 无答案：论文里确实没写，且能用 `grep` 证明。每条在数据集里附 `absence_proof`（验证命令 + 命中数 0）。

## 3. 状态契约

一次运行的全部中间结果。字段名固定，因为测试要断言它们。

| 字段 | 含义 | 写入方 | 读取方 |
|---|---|---|---|
| `question` | 原始问题（永不改写） | 入口 | 全部节点 |
| `strategy` | `simple` / `complex` | route | 条件边、评测 |
| `route_reason` | 路由理由，一句话 | route | trace |
| `sub_questions` | 有序子问题列表，1–6 条 | decompose | retrieve、plan |
| `next_idx` | 下一条待检索子问题的下标 | decompose(置 0)、retrieve(自增) | plan、条件边 |
| `current_query` | 本轮实际送去检索的查询 | retrieve | trace |
| `rounds` | 已完成检索轮数 | retrieve(自增) | plan、条件边 |
| `max_rounds` | 本次运行的轮数上限 | 入口（来自配置） | plan、条件边 |
| `candidates` | 累积候选，按 `chunk_id` 去重 | retrieve | plan、fuse |
| `fused` | RRF 融合后的排序结果 | fuse | rerank、answer 降级 |
| `selected` | 最终进入上下文的片段（≤ `top_n`） | rerank（关闭时由 fuse 直通） | answer、评测 |
| `context` | 拼好的上下文字符串 | answer 前 | answer |
| `answer` | 最终回答 | answer | 出口、评测 |
| `trace` | 逐节点的 `{节点, 耗时ms, 决策, 细节}` | 每个节点 | CLI `--verbose`、报告 |

**消费关系必须唯一**：`answer` 只读 `selected`；`selected` 为空时退到 `fused[:top_n]`；两者皆空则走拒答分支。不允许出现「写了但没人读」的字段。

## 4. 行为条款

### B1 · 路由（simple 不检索）
- **Given** 问题不需要语料中的具体事实（算术、通用定义、寒暄、与产品无关的常识）
- **When** 运行入口
- **Then** `strategy=simple`，直接作答，**不产生任何检索调用**，`selected` 为空，`trace` 只有 route 与 answer 两条。

### B2 · 路由（complex 必拆解）
- **Given** 问题需要具体参数、流程、角色、时限、编号，或需要用两处以上事实推理
- **When** 运行入口
- **Then** `strategy=complex`，进入 decompose；**不允许**跳过拆解直接检索。

### B3 · 拆解约束
- **When** 进入 decompose
- **Then** 输出 1–6 条**有序**子问题，且同时满足：
  1. 每条是完整英文问句，可独立检索；
  2. 不含「他 / 她 / 此人 / 该套餐 / 上述」等依赖上下文的指代，实体必须写全；
  3. 顺序符合推理链：先定实体，再查属性；
  4. 不得把原题原样复制成唯一一条（除非确实无法拆分）。
- **异常**：模型返回超过 6 条 → 保序截断为 6；返回 0 条或全为空串 → 降级为 `[question]` 一条，并在 `trace` 标记 `decompose_degraded`。降级不抛弃本轮，继续检索。

### B4 · 混合召回
- **When** 有 `current_query`
- **Then** 同时做两路召回（语义向量 + 关键词），各自取 `top_k_each`；两路结果按**统一 id 去重**后并入 `candidates`；同一个 id 出现两次时保留两路分数，供融合阶段使用。
- **约束**：两路必须独立发起（一路失败不影响另一路），任一路失败时按「单路召回」继续，并在 `trace` 标记 `retrieval_degraded`。

### B5 · 规划与熔断
- **When** 一轮检索结束
- **Then** 由规划步骤决定下一步是「再检索」还是「生成」。规划只能二选一，不得自拟新查询。
- **代码层两道硬熔断**（优先于模型判断，必须生效）：
  1. `rounds >= max_rounds` → 强制生成；
  2. `next_idx >= len(sub_questions)` → 强制生成。
- **验收**：规划持续选择「再检索」且 `max_rounds=2` 时，检索恰好执行 2 次，第 3 次不得发生。

### B6 · 空召回降级
- **Given** 本轮或累计召回为空
- **When** 即将生成
- **Then** **不得**把空上下文送入生成。必须走拒答分支：`answer` 以 `I cannot confirm this from the provided sources` 开头，并附「缺什么」的一句说明（来自最近一轮的 `current_query`）。
- **断言**：召回为空时，生成模型的调用次数为 0。

### B7 · 作答与拒答
- **When** 有 `selected`
- **Then** 回答中的每一条事实性论断都必须能在 `selected` 中找到依据；不得引入语料之外的信息。
- **When** 上下文不足以确定关键事实
- **Then** 首句固定为 `I cannot confirm this from the provided sources`，随后说明缺哪一项，**不得**猜测或补全。
- **理由**：固定标记串让拒答判定可以走规则而不是判官，稳定且零成本（见 ADR-005）。代价是措辞略僵硬，接受。

### B8 · 错误转结构化
- **When** 检索层、模型层或配置层出错
- **Then** 转成三类可区分的错误（检索失败 / 模型失败 / 配置无效），返回可读提示并以非零退出码结束；**不允许**抛未捕获异常，**不允许**退出码为 0。
- **断言**：注入一次检索网络异常，进程退出码非 0，stderr 含「检索」字样。

## 5. 评测契约

### 数据集（30 条，`evals/dataset.jsonl`）

```json
{"id":"q01","question":"...","reference_answer":"...","relevant_chunk_ids":["c12","c31"],
 "type":"single_hop|multi_hop|unanswerable","expected_strategy":"simple|complex",
 "absence_proof":"grep 命令 + 命中数（仅 unanswerable）"}
```

- 配比：单跳 14、多跳 10、无答案 6。
- **问题与参考答案一律英文**，事实必须能在 `data/corpus/` 原文中逐字定位；参考答案里出现的每个数字都要能在标注片段里找到。
- `relevant_chunk_ids` 是标注的**正确片段**（可能 1–3 个），`context_recall` 的分母。
- 多跳题的 `relevant_chunk_ids` 必须**跨至少两个不同 `source` 文件**，否则降级为单跳（ADR-014）。
- `expected_strategy` 用于路由准确率；单跳题也可以是 `complex`（需要具体数字），**允许标注与直觉不同**，但必须在数据集注释里说明判据。
- 无答案题的 `relevant_chunk_ids` 填「语义上最接近但不含答案」的片段——这样召回高而拒答错时，能区分「召回没问题、是模型在编」。`absence_proof` 记录当时的 grep 命令与 0 命中结果，任何人可复核。

### 五个指标

| 指标 | 判定方式 | 定义 | 分母 |
|---|---|---|---|
| `context_recall` | 确定性 | 标注片段中被 `selected` 覆盖的比例 | 28 条：`relevant_chunk_ids` 非空的样本 |
| `groundedness` | 判官 | 每个论断能否在 `context` 找到依据，0–1 | 全部 30 条 |
| `answer_relevance` | 判官 | 是否正面回答了问题，0–1 | 全部 30 条 |
| `refusal_accuracy` | 规则 + 判官复核 | 无答案题是否正确拒答（首句含固定标记） | 仅 6 条 `unanswerable` |
| `routing_accuracy` | 确定性 | `strategy` 与 `expected_strategy` 一致的比例 | 全部 30 条 |

判官输出必须为结构化 `{score, reason, unsupported_claims[]}`。解析失败重试 1 次；仍失败记 `null`，**单列 `null_rate`，绝不当 0 或 1**。

`context_recall` 的分母口径（ADR-020）：2 条 `simple` 题的 `relevant_chunk_ids` 就是空列表——它们测路由，不测召回。**排除条件看「标注」而不是「实际判成什么」**：某题标注为 `complex` 却被判成 `simple`（因此不检索），`selected` 为空 → recall 记 0 并留在分母里。若按「实际判成 simple 就排除」来算，误判反而会让指标变好看，等于奖励偷懒。

### 门禁（阈值写死在 `config.yaml`）

| 项 | 阈值 | 说明 |
|---|---|---|
| `context_recall` | ≥ 0.70 | |
| `groundedness` | ≥ 0.80 | |
| `answer_relevance` | ≥ 0.80 | |
| `refusal_accuracy` | ≥ 0.80 | 6 条里至少 5 条 |
| `routing_accuracy` | ≥ 0.85 | 30 条里至多错 4 条 |
| `null_rate` | ≤ 0.10 | 判官可靠性闸；超了即便分数高也 fail |

`--gate` 时任一不达标 → 退出码 1，并明确打印「差在哪个指标、差多少、距离阈值几个样本」。

### 报告

- `reports/{ts}-eval.json`：逐条原始结果（含 trace 摘要、token 用量、各节点耗时）。
- `reports/{ts}-eval.md`：汇总表、失败样本 top5（按失分排序）、每节点平均耗时、token 总量。
- 带 `--baseline` 时追加 `指标 | 基线 | 本次 | 变化` 表；`--baseline` 缺省取 `reports/` 里最新一份**普通评测** JSON（跳过 `-ablation`，它的结构不同）。
- `reports/{ts}-ablation.json` / `.md`：四组消融，每组只比上一组多一个开关，Δ 相对第一组。
- `reports/` 本身不进版本库（`.gitignore`）：报告可由命令重放，仓库只存能重放它的代码。

## 6. CLI 契约

| 命令 | 行为 |
|---|---|
| `python -m app.cli --ask "问题"` | 打印回答。**首次运行**若索引不存在，自动构建后回答（一条命令可跑通是硬要求） |
| `--verbose` | 追加打印每个节点的耗时与决策理由，以及最终 `trace` |
| `--reindex` | 强制重建索引 |
| `python -m evals.run` | 跑全量 30 条，产出两份报告，退出码 0 |
| `python -m evals.run --gate` | 额外执行门禁，不达标退出码 1 |
| `python -m evals.run --limit 5` | 分片跑，用于调试判官（门禁在 `--limit` 下拒绝执行，防误判） |

## 7. 验收标准（必须贴真实输出，跑不了就写明卡在哪）

1. `pytest -q` → 5 个用例全绿，且**全程不联网、不消耗 token**（假模型）。
2. `python -m app.cli --ask "..."` → 能看到节点顺序与耗时。
3. `python -m evals.run --gate` → 产出报告，退出码 0。
4. 核心逻辑行数（`app/` 下，`wc -l`）落在 600–800。
5. `plan` 返回「再检索」时图确实回到检索：指出测试文件与行号。

## 8. 非功能约束

- Python 3.11+，全量类型注解，跨节点数据用 `pydantic` 模型。
- 依赖 ≤ 8 个，每个在 README 写「为什么需要」。
- 默认路径不依赖任何外部中间件、不联网（除模型 API）。
- 不接入任何云端评测平台，结果只落本地文件。
