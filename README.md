# Agentic RAG：带路由、多跳规划与离线评测的检索问答

在 14 篇 RAG 顶会论文（NeurIPS / ICLR / ACL / EMNLP / NAACL / TACL）上做检索问答。
问题先判「要不要查资料」，需要查就拆子问题、边查边规划，答不出来就明说缺什么——
每一步都能在 trace 里看到耗时与决策。

**这不是从零另写的一个项目**，而是在公开 Agentic RAG 实现（`ai_doubao_ysw` 仓库的
`advanced-rag` 与 `rag-demo`）基础上做的融合重构：保留它的三段式编排与五版演进思路，
修正其中 15 个会让评测失真的具体缺陷（见 [§5](#5-与源实现的关系)），并把评测层重写为
可离线、可复现、带门禁的形态。

```
app/     核心逻辑，750 代码行（物理 1187，其余为注释、docstring 与空行）
evals/   评测层：30 条标注集 + 3 个判官 + 门禁 + 消融（物理 705）
tests/   37 个离线回归断言，假模型，约 4 秒（物理 656）
data/    语料抓取脚本 + 哈希锁定清单（全文不入库）
```

---

## 1. 三条命令

```bash
pip install -e ".[dev]"
python scripts/fetch_corpus.py          # 按 corpus.lock.json 抓 14 篇论文，校验 sha256
python -m app.cli --ask "How many QA pairs does CRAG contain?" --verbose
```

首次 `--ask` 会自动建索引（1,357 个片段）。`--verbose` 按下面的格式打印每个节点的耗时
与决策——**这是格式示意，毫秒数是占位值**，真实耗时需要填 key 后跑一次（§4）：

```
route         312ms  complex  需要具体数值，须查语料
decompose     890ms  2 条     CRAG 的规模与领域数需要分两次检索
retrieve       18ms  CRAG question-answer pairs  召回 8 条，累计 8 条
plan          402ms  retrieve  第 2 条子问题尚未检索
retrieve       16ms  CRAG benchmark domains      召回 8 条，累计 14 条
plan          388ms  generate  证据已足够
answer       1204ms  5 片段
```

评测：

```bash
python -m evals.validate                       # 先校验数据集本身（7 项检查）
python -m evals.run --limit 10                 # 小样本冒烟
python -m evals.run --gate --baseline reports/xxx.json
python -m evals.run --ablation --limit 10      # 四组消融
pytest -q                                      # 37 个断言，不联网、不耗 token
```

`pytest` 在**未抓语料**的干净克隆上会有 3 条跳过（跳过的理由直接打印「先跑
scripts/fetch_corpus.py」），其余 34 条照常通过——测试不该因为缺数据就假装失败。

报告落在 `reports/{ts}-eval.{json,md}` 与 `reports/{ts}-ablation.{json,md}`。**`reports/` 不进版本库**——
报告能由命令重放，仓库只存能重放它的代码。`--baseline` 缺省取 `reports/` 里最新一份普通评测，
跳过 `-ablation`（两者结构不同），并会在 embedder 变化时警告 "Δ 不可比"。
基线路径打错时**开跑前就退出 2**，不会烧完几百次判官调用才告诉你路径错了。

---

## 2. 状态图

```
                    ┌─────────┐
       question ───▶│  route  │  判 simple / complex
                    └────┬────┘
             simple ────┘ └──── complex
                │                │
                │         ┌──────▼──────┐
                │         │  decompose  │  拆 1–6 条有序子问题
                │         └──────┬──────┘
                │                │
                │         ┌──────▼──────┐
                │         │  retrieve   │◀──────┐
                │         └──────┬──────┘       │
                │                │              │
                │         ┌──────▼──────┐       │
                │         │    plan     │───────┘  next_action == retrieve
                │         └──────┬──────┘          （代码层两道熔断先于模型）
                │                │ generate
                └────────────────▼
                          ┌──────────┐
                          │  answer  │  空召回 → 拒答，生成调用 = 0
                          └──────────┘
```

**回边的条件边只读 `state.next_action`，而这个字段只有 `plan` 会写。**
这是刻意的：源实现的 `afterPlan` 读的是 `route` 阶段写入的 `strategy`（恒为 `complex`），
于是「多跳回边」从未执行过一次，而日志里 `plannedNext` 打得漂漂亮亮，谁也不会怀疑。
把「条件边读的字段」与「节点写的字段」在类型上变成同一个，读错就立刻报错而不是静默降级。

---

## 3. 五个设计决定

| 决定 | 为什么不选更省事的做法 |
|---|---|
| **简单题不检索**（`route` 判 `simple` 直接答） | 算术、定义、寒暄去检索只会引入噪声。代价是路由本身成为指标的一部分，所以单设 `routing_accuracy`，而不是让 `context_recall` 的分母为零 |
| **双路召回 + RRF，不按分数加权** | 余弦在 0–1、BM25 无上界且随语料规模漂移，任何加权系数都是拍脑袋。RRF 只用名次，与量纲无关 |
| **numpy 精确余弦，不引 Milvus/ES** | 1,357 片段下一次精确检索是毫秒级且**逐位可复现**；ANN 在这个量级只带来近似误差。零中间件才能让 `git clone` 后一条命令跑通，`--baseline` 的 diff 才有意义 |
| **拒答用固定首句** | 让拒答可被规则判定，成本为零且不抖动。拒答是格式问题，不是语义品味问题；判官只用于复核「标记存在但仍在编」 |
| **判官失败记 `null` 而不是 0 分** | 记 0 会把「API 抖了」显示成「模型变差了」。`null_rate` 因此是独立指标，且是反向门禁（>10% 直接 fail），而不是把 null 悄悄剔掉 |

---

## 4. 评测

### 指标

| 指标 | 怎么算 | 需要 LLM 吗 |
|---|---|---|
| `context_recall` | 标注片段被 `selected` 覆盖的比例 | 否（确定性） |
| `routing_accuracy` | 实际 `strategy` 与标注是否一致 | 否（确定性） |
| `refusal_accuracy` | 该拒的拒了、不该拒的没拒 | 规则为主，判官复核 |
| `groundedness` | 答案的每条论断是否被片段支持（0/1/2 三档归一） | 是 |
| `answer_relevance` | 答案是否真的回答了问题（0/1/2 三档归一） | 是 |
| `null_rate` | 判官调用失败占比（反向指标） | — |

`context_recall` 只在「确实该检索」的样本上计算，**分母是 28 条**（2 条 `simple` 题的标注为空）。
排除条件看**标注**而不是实际路由：标注 `complex` 却被错判成 `simple`、因此没检索的样本，
recall 记 0 并留在分母里——否则误判会把样本挤出分母，指标反而变好看，等于奖励偷懒（ADR-020）。

### 数据集

30 条，英文，事实逐条从论文原文取：单跳 14 / 多跳 10 / 无答案 6。

- **多跳题的标注片段强制跨 ≥2 个 source**，由校验脚本断言。不跨就是单跳，不能算多跳。
- **参考答案里的每个阿拉伯数字必须在标注片段中逐字出现**，由校验脚本断言。
  这条拦住的是「凭记忆写数字」——我自己就这么写错过一次（把 21 点记成 20 点）。
- **无答案题带 `absence_proof`**：存正则与作用域，当场重跑必须是 0 命中。
  六条题分别考「短语级缺失」与「归属错误」两类陷阱，例如：
  - *CRAG 报告的标注者间一致性是多少* —— 语料里只有 RAGAS 报了 95%，检索会命中，但归属不对
  - *RAGChecker 每次评估查询的 API 成本* —— RAGChecker 真报了成本，但报的是人工标注报酬（15 USD/小时），答 255 就是错

### 当前状态

`index.embedder=hash` 是**占位向量**（确定性词袋哈希），只用于验证「切块 → 落盘 → 矩阵乘 →
top-K → RRF」这条链路是通的。它没有语义泛化能力，**用它跑出的 `context_recall` 不能写进
任何结论**，报告头会打印显式警告。

填入真实 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 后：

```bash
# config.yaml: index.embedder: openai
python -m app.cli --reindex
python -m evals.run --gate
```

| 指标 | 阈值 | hash 占位 | openai 真实 |
|---|---|---|---|
| context_recall | 0.70 | 待填 | 待填 |
| groundedness | 0.80 | 待填 | 待填 |
| answer_relevance | 0.80 | 待填 | 待填 |
| refusal_accuracy | 0.80 | 待填 | 待填 |
| routing_accuracy | 0.85 | 待填 | 待填 |
| null_rate | ≤0.10 | 待填 | 待填 |

> 阈值不在这里拍脑袋定：第一篇真实报告出来之后才校准，并把调整理由写进
> `docs/decisions.md`。现在这一列是空的，是因为它还没跑过——不是忘了填。

### 消融

`--ablation` 每次只加一个开关，四组同数据集：

| 变体 | 改了什么 |
|---|---|
| A 单路向量 | `hybrid=false`，`multihop=false` |
| B +BM25（拼接） | `hybrid=true`，`fusion=concat`——**这是源实现的形态**：两路结果直接拼、按 id 去重，向量相似度不参与排序 |
| C +RRF | `fusion=rrf`，名次融合 |
| D +多跳 | `multihop=true`，回边与两级熔断挂上 |

B 组不是凑数的：它把「源实现停在 hybrid 版却没做到融合」变成一个可测的基线，
「RRF 值多少个百分点」才有据可依。

---

## 5. 与源实现的关系

源仓库有五个渐进版本（`naive-rag` → `query-router` → `multihop` → `webfallback` →
`hybrid`），逐版加能力，但每一版都有会让**评测失真而不报错**的问题。融合时逐条处理：

| 源实现 | 这里的处理 |
|---|---|
| `afterPlan` 读 `strategy`，回边永不触发 | 条件边只读 `next_action`，`test_loop_guard` 钉住「检索恰好 2 次」 |
| `retrievedDocs` / `retrieveDocs` 字段名错位，LangGraph 静默丢弃 | `State` 用 `extra="forbid"`，写错字段当场 `ValidationError` |
| 空召回检查写在生成之后（先编再道歉） | 拒答在 `answer` 入口，生成调用次数断言为 0 |
| `hybrid` 只做 `concat + 去重`，不算 RRF | `rrf_fuse` 按名次融合，两路原始分各自保留 |
| 三种收尾写法（`.catch(console.error)` / `.then(err=>)` / 裸 `main()`）退出码全为 0 | 三类错误映射三种退出码，`test_error_path` 断言互不相同且非零 |
| `catch` 未绑定变量却引用 `error.message` | 包装异常时保留 `__cause__`，原始信息透出 |
| Milvus 与查询端度量不一致 | 索引与查询共用同一 embedder，`manifest` 记 `embedder` 与 `tokenizer` |
| 无产物、无门禁，跑完只打印一行 | `reports/*.json` 落盘 + `--gate` 退出码 + `--baseline` diff |

被放弃的能力也写清楚了：**联网兜底**（会让离线评测不可复现，且 6 条无答案题会被搜索直接答掉）、
**replan**（状态机多一条回边、测试翻倍，且源实现没有可参照的正确形态）、
**Milvus / ES / LangSmith**（见上表第三章）。

---

## 6. 已知限制

1. **语料只有 1,357 个片段，`top_k=8` 覆盖 0.6%**。这个量级下精确检索优于 ANN，但代码没有
   在千级以上验证过；扩展点（换 ANN、分批 embedding）写了，没跑过。
2. **不做 replan**。拆解质量就是上限：多跳题若停在 0.5，第一嫌疑是 `decompose` 的 prompt，
   不是检索。
3. **标注取「最直接的那一片段」**。同一事实在摘要与正文重复出现时只标最权威的一处
   （摘要单独成块时会补标），所以 `context_recall` 的绝对值天然偏低，跨系统比较才有意义。
4. **`hash` 占位向量下的任何指标都不可用**，见 §4。
5. **中英混排未处理**。语料与问句都是英文；换中文语料要重做 tokenizer（当前是空格切分 +
   保留数字与 `BM25` 这类词形），不能直接沿用。

---

## 7. 行数口径

| 部分 | 代码行 | 物理行 | 说明 |
|---|---:|---:|---|
| `app/` | 750 | 1,187 | 代码行用 `tokenize` 排除注释、docstring、空行 |
| `evals/` | — | 576 | 评测层，与核心逻辑分开计 |
| `tests/` | — | 630 | 35 个断言，3.6 秒，不联网 |

两个数都报，是因为只报代码行不诚实（`wc -l` 一跑就对不上），只报物理行则会逼着删掉
「每条注释指认一个源缺陷」这个特性。核心逻辑 750 行落在 600–800 的预算内。

---

## 8. 文档

| 文件 | 内容 |
|---|---|
| `docs/spec.md` | 行为契约（Given/When/Then），唯一的「做完了没」标准 |
| `docs/decisions.md` | 17 条裁定：问题 / 选项 / 裁定 / 代价 |
| `docs/plan.md` | 架构、行数预算、来源映射（18 项能力的去向） |
| `docs/tasks.md` | 15 个任务与完成判据、缺陷→测试对齐表 |
| `docs/source-audit.md` | 源仓库勘察：15 条带行号的缺陷 |
| `docs/resume.md` | 简历叙事与「数字来源对照表」（跑不出来的数字留空） |
