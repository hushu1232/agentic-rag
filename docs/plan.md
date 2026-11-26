# Plan · 实现方案

> 输入：`spec.md`（行为）、`decisions.md`（裁定）、`source-audit.md`（源仓库勘察）。
> 本文件含原「阶段 1 映射表」全部内容，不再单独出 `merge-mapping.md`。

## 1. 分层

```
cli.py            入口：解析 --ask/--verbose/--reindex，拼 Deps，调 graph，打印 trace
  └ graph.py      状态图：5 节点 + 3 条条件边（不含业务逻辑，只连边）
      └ nodes.py  5 个节点函数：route / decompose / retrieve / plan / answer
          ├ retrieve.py   两路召回（vector + bm25）→ RRF 融合
          ├ rerank.py     重排接口（默认关闭时直通）
          ├ index.py      切块、embedding、npz 存取、BM25 建库
          ├ llm.py        唯一的模型出口：重试、超时、token 统计、结构化解析
          ├ schemas.py    State + 所有 pydantic 模型（跨节点契约）
          └ config.py     config.yaml + 环境变量 → 带校验的 Config
```

依赖方向单向向下，**`nodes.py` 不认识 `cli.py`，`graph.py` 不认识 `llm.py`**。这条约束的作用：所有外部调用必须经过 `llm.py` / `index.py`，测试里替换这两个就够了。

## 2. 数据流（一次 complex 提问）

```
question
  │
  ├─ route ────────────→ {strategy, route_reason}          [1 次模型调用]
  │     └ simple ──────────────────────────────────→ answer ─→ END
  │
  ├─ decompose ─────────→ sub_questions[1..6], next_idx=0  [1 次模型调用]
  │
  ├─ retrieve ◄──────────────────────────┐                 [1 次 embedding + 2 次检索]
  │     current_query = sub_questions[next_idx]
  │     vector: cosine(query, npz) top_k_each
  │     bm25:   bm25.get_top_n(query) top_k_each
  │     → candidates (按 chunk_id 去重，保留两路分数)
  │     → fused = RRF(candidates) → selected = rerank(fused)[:top_n]
  │     rounds += 1, next_idx += 1
  │
  ├─ plan ──────────────→ {next_action, reason}            [1 次模型调用]
  │     硬熔断：rounds>=max_rounds 或 next_idx>=len(subs) → generate
  │     └ retrieve ──────────────────────┘
  │     └ generate
  │
  └─ answer ────────────→ answer                           [1 次模型调用]
        selected 空 → 拒答分支（0 次生成调用）
```

模型调用上限：`1 + 1 + max_rounds×1 + 1`。`max_rounds=3` 时最多 6 次。这是成本可预算的原因，也是评测跑 30 条不会失控的原因。

## 3. 索引（无中间件）

| 项 | 实现 | 落盘 |
|---|---|---|
| 切块 | 按 `## ` 标题切节，节内按 `chunk_size=800` 定长切、重叠 100，携带 `source` / `heading` / `index` → `chunk_id = "{source}#{index}"` | —— |
| 向量 | 批量 embedding → `float32` 矩阵 | `.index/vectors.npz` + `.index/chunks.jsonl` |
| 检索 | `L2 归一化后矩阵乘` = 余弦相似度，`np.argpartition` 取 top-k。**精确检索**，不引 ANN | 内存 |
| 关键词 | `rank_bm25.BM25Okapi`，英文按**小写 + 去标点 + 空白切分**建 token，保留数字与 `recall@k` 这类词形（ADR-006 修订） | `.index/bm25.pkl` |
| 失效 | `.index/manifest.json` 存语料文件的 `mtime` 与 `sha1` 汇总；任一文件变化即自动重建；另存 `corpus_lock_sha` 校验抓到的语料与 `data/corpus.lock.json` 一致 | —— |

实测语料 833,362 字，切块后 **1,357 个片段**（单篇 40–149，平均 665 字）。一次精确检索是 O(N·d) ≈ 1357×1536 次乘加，仍在毫秒级——**精确检索在这个量级依然是正收益**，这是 spec 里「不要引 Milvus」的技术依据。片段规模从原方案的百级涨到千级，是 ADR-014 换语料的直接后果：`top_k=8` 覆盖 1357 片段的 0.6%，比原来更难，`context_recall` 阈值需按真实基线定（T13）。

## 4. 行数预算

| 文件 | 预算 | 实际代码行 | 实际物理行 | 内容 |
|---|---:|---:|---:|---|
| `config.py` | 145 | 111 | 157 | 六段配置的显式模型 + `extra="forbid"` + gates 完整性/区间校验 + `.env` 解析 |
| `schemas.py` | 60 | 58 | 105 | `State`、三个结构化输出、`Chunk`/`Candidate`/`TraceStep`/`RunResult` |
| `llm.py` | 80 | 71 | 113 | 重试、超时、`complete` / `complete_structured`、token 累计、`LlmProto` |
| `index.py` | 95 | 164 | 274 | 切块 + 两种 embedder + BM25 + manifest 失效校验（**超预算的主因**） |
| `retrieve.py` | 70 | 67 | 111 | 双路检索 + RRF + 降级 + 多 query 入口 |
| `rerank.py` | 30 | 41 | 67 | HTTP 重排客户端 + 关闭态直通 + 下标校验 |
| `nodes.py` | 170 | 140 | 195 | 五节点 + 四段 prompt + 两道硬熔断 |
| `graph.py` | 50 | 29 | 56 | 建图、三条条件边、`multihop=false` 短路 |
| `cli.py` | 50 | 51 | 77 | argparse、verbose trace、三类错误退出码 |
| **合计** | **750** | **732** | **1156** | 代码行在 600–800 内；物理行超预算，原因见 ADR-016 |

> **预算修订记录（T01 实测）**：`config.py` 原估 40 行，实际 147 行——要做到「yaml 拼错 / gates 缺项 / 阈值越界 / 缺环境变量」四类问题都在启动时报错，必须显式声明六段配置模型，40 行做不到，原估偏乐观。差额从 `graph.py`（原 70→50）、`cli.py`（原 60→50）、`rerank.py`（原 40→30）三处过度预留里挪出，总额 740→750，仍在 600–800 内。**下一次修订只允许在 T10 结束时做**，避免边写边改预算。

`evals/`（第二预算）：`judges.py 90` / `run.py 150` / `report.py 90` = 330（比原定 290 多 40 行，用于 ADR-012 的消融循环）。

## 5. 依赖（7 个）

| 依赖 | 为什么需要 |
|---|---|
| `langgraph` | 题目指定用它的 StateGraph 手写图；条件边与循环是核心（自己写状态机要 150 行且不可避免要测回边） |
| `pydantic` | 跨节点数据结构 + 模型结构化输出的解析校验；`dict` 会让 `retrieveDocs` 这类字段名错到运行期才发现（源仓库 D02 就是这个问题） |
| `openai` | 唯一的模型出口，OpenAI 兼容协议；换供应商只改 `base_url` |
| `numpy` | 精确余弦与 RRF 排序；纯 Python 循环在 120×1024 上慢一个量级 |
| `rank_bm25` | 关键词侧现成实现，约 60 行；自己写 IDF 与打分的价值为零 |
| `pyyaml` | `config.yaml`（阈值、取数、轮数上限都在这里，改参数不改代码） |
| `pytest` | 5 个回归测试；dev 依赖 |

**不引入**：`python-dotenv`（`config.py` 里 6 行读 `.env` 即可）、`jieba`（ADR-006）、`tenacity`（`llm.py` 里 10 行重试）、`openevals`（判官自己写，见 §7）、`tiktoken`（token 统计直接用 API 返回的 usage）。

## 6. 来源映射

> **本节是 ADR-013「融合而非重建」的落地清单**。`§6.0` 把 13 个文件拆成 18 项能力逐条判定去向，`§6.1` 逐文件写保留/改写/丢弃。任何一项能力没有归宿，就是跑偏。

### 6.0 融合清单（18 项能力，每项必有判定）

| # | 能力 | 源出处 | 判定 | 落点 |
|---:|---|---|---|---|
| 1 | 线性「检索→生成」 | `naive-rag` | 合并 | `nodes.retrieve` → `nodes.answer`（作为 complex 的退化形态） |
| 2 | 问题路由 simple/complex | `rag-query-router:43-71` | 保留 | `nodes.route` |
| 3 | 有序子问题拆解 + 禁指代词 | `rag-multihop:133-172` | 保留（上限 8→6，截断而非抛错） | `nodes.decompose` |
| 4 | 跨轮累积去重 | `rag-multihop:174-226` | 保留（去重键改为 `chunk_id`） | `nodes.retrieve` + `retrieve.py` |
| 5 | 规划二选一 + 两道硬熔断 | `rag-multihop:228-287` | 保留（修正条件边读错字段） | `nodes.plan` + `graph.py` |
| 6 | 信息充分性评估（`enough`/`missing`） | `rag-webfallback:132-167` | 合并 | `nodes.answer` 的拒答分支：`missing` 用于解释「缺哪一项」 |
| 7 | 联网搜索兜底 | `rag-webfallback:222-300` | **弃用**（ADR-003：评测须离线可复现） | — |
| 8 | 查询扩展（3 条改写） | `query-augment` | 保留，默认关闭 | `retrieve.py` 入口接受**多条 query**；`retrieval.query_augment` 开关 |
| 9 | 双路召回（文本 + 向量） | `hybrid-retrieval:162-213` | 合并（ES → BM25 内存） | `retrieve.py` |
| 10 | 按 id 去重、同 id 保留先到者 | `hybrid-retrieval:71-89` | 保留思路，**改写**：保留两路分数而非先到者 | `retrieve.py` |
| 11 | 重排（cross-encoder） | `dashscope-rerank` | 保留为可开关 | `rerank.py` |
| 12 | 有/无上下文两套 prompt | `hybrid-retrieval:118-148` | 保留 | `nodes.answer` |
| 13 | 双库同源写入、字段对齐 | `seed-data:122-263` | 合并为**单索引**（npz + bm25 + manifest） | `index.py` |
| 14 | 切块参数 500/50 | `milvus_insert:30-53` | 保留参数，**改写**：先按标题切再定长，补 `chunk_id` | `index.py` |
| 15 | 12 条 FAQ 评测集 | `build_dataset:6-57` | 合并（保留题面风格）→ 30 条三类 | `evals/dataset.jsonl` |
| 16 | 三个判官的分工与入参映射 | `evaluators:18-62` | 保留分工，重写实现（结构化 + 重试 + `null`） | `evals/judges.py` |
| 17 | 批量评测适配层 | `run_eval:13-29` | 保留思路（`answer + context` 进评测） | `evals/run.py` |
| 18 | Milvus / ES / LangSmith 云 / openevals | 分散在多文件 | **弃用**（替代方案与指标含义变化见 §8） | — |

**唯一新增的判定**：#8 查询扩展在原计划里被我整段丢弃，理由是「与 `decompose` 重复」。按 ADR-013 第 3 条，弃用必须有理由且经确认——而它其实可以零成本融合：`retrieve.py` 的入口本来就该接受「一组 query」（源实现的 hybrid 就是拿 4 条改写并行召回），把单条泛化成多条只多约 12 行。所以改为**保留但默认关闭**，并顺手成为消融的第五个维度。

### 6.1 逐文件映射

| 源位置 | 目标文件 | 保留什么逻辑 | 改成什么 | 丢弃什么 |
|---|---|---|---|---|
| `naive-rag.mjs:18-23, 59-107` | `schemas.py` `nodes.py` | State 字段集合；「片段 N / 章节 / 内容」拼装模板 | `Annotation.Root` → pydantic `State`；模板进 `config.yaml` | 固定问题、Milvus 连接、`process.stdout` 流式打印 |
| `rag-query-router.mjs:43-71, 165-182` | `nodes.py(route)` `graph.py` | `RouteSchema` 二分类 + `decideNext` 条件边形态 | zod → pydantic；`route_reason` 真正被消费（写进 trace） | `console.log`、`state.k` 一路透传（k 进配置） |
| `rag-query-router.mjs:73-96` | `nodes.py(answer)` | 直答 prompt | 与 RAG 作答合并为一个 `answer` 节点，按 `selected` 是否为空分两支 | 独立的 `direct_answer` 节点（省 20 行，行为不变） |
| `rag-multihop.mjs:133-172` | `nodes.py(decompose)` | 有序子问题、禁指代词、须可独立检索；`trim+filter` 后判空 | 1–8 → 1–6；`throw` → 保序截断 / 降级为 `[question]` 并在 trace 标记 | `console.log` 逐条打印 |
| `rag-multihop.mjs:174-226` | `nodes.py(retrieve)` `retrieve.py` | 按 `next_idx` 推进；跨轮累积去重（避免重复片段造成"多方证据"错觉） | 单路 Milvus → 双路 + RRF；去重键 `chunk_id`（不是可空 id） | `String(d.id)` + `"unknown"` 兜底（D07）、`throw Error` 中断 |
| `rag-multihop.mjs:228-287` | `nodes.py(plan)` | **两道硬熔断**（轮数 / 子问题耗尽）——源仓库唯一写对的部分，逐条保留；prompt 里「下一轮自动使用下一条子问题，你不要自拟查询」的措辞 | 条件边读 `next_action` 而非 `strategy`（**D01 的正解**） | `afterPlan` 原函数（就是它读错字段） |
| `rag-multihop.mjs:289-327` | `nodes.py(answer)` | 五条回答要求（只依据片段、可综合、无依据就说、可引用原文） | 追加拒答固定标记；上下文改读 `selected` | 流式打印（CLI 层做） |
| `rag-webfallback.mjs:132-167` | `nodes.py(answer)` | 「信息充分性」的判定维度（`enough` / `missing`） | 折进 `answer` 的拒答说明：`missing` 用来解释「缺哪一项」 | `web_query` 字段、二次评估的整条回边（ADR-003） |
| `rag-webfallback.mjs:222-300` | —— | —— | —— | **整段丢弃**：博查搜索、`catch` 引用未定义 `error`、网页结果判空取反 |
| `hybrid-retrieval.mjs:57-89` | `retrieve.py` | 不同来源文档的 metadata 归一化；按 id 去重 | `chunk_id` 作键；同 id 保留**两路分数**而不是「先到者」 | ES 优先覆盖（D08）、`id` 可空的假设 |
| `hybrid-retrieval.mjs:162-251` | `graph.py` `retrieve.py` `rerank.py` | 双路并行召回 + fan-in 汇合边的形态；`formatDocsAsContext` 的 `[i] id= source=` 格式 | 并行改 `asyncio.gather`；汇合后接 RRF；rerank 做成可开关 | `!merge.length`（函数取 .length，D05）、硬编码 `INDEX`/`localhost`、`console.log(batches)` |
| `rag/query-augment.mjs:33-49` | —— | —— | —— | **丢弃**：本项目的「查询扩展」由 `decompose` 承担；源实现的 3 条改写在多跳场景下反而放大噪声，且降级时用原句补齐导致同一 query 检索 4 次（D19） |
| `rerank/dashscope-rerank.mjs` | `rerank.py` | 按 `output.results[].index` 回取原文档的外壳思路 | 继承 LangChain 基类 → 30 行 `httpx`/`urllib` 客户端；`enabled=false` 直接返回入参 | 类继承、`res.json()` 先于 `res.ok`（D14）、下标不校验 |
| `rag/seed-data.mjs:122-263` | `index.py` | 「同源数据双写两套索引」的思路（两库字段对齐的必要性） | 落地为 `vectors.npz` + `bm25.pkl` 一份 manifest 管两者 | ES/Milvus 客户端、`priority` 类型不一致（D16）、L2 建索引 vs COSINE 检索（D09） |
| `milvus_insert.mjs:30-53` | `index.py` | 切块参数 `chunkSize=500 / overlap=50` 的取值经验 | 先按标题切、再定长切；metadata 增加 `chunk_id` | 「只有 `source` 没有切片 id」、`console.log(chunks/vectors)` |
| `rag-demo/src/rag_agent.mjs:80-86` | `cli.py` | `ask(q) → {answer, context}` 这个**评测友好**的接口形状 | 返回 `RunResult`（含 trace、candidates 数、token） | 模块级 Milvus 连接、`state.context` 传 Document 对象进状态 |
| `eval/build_dataset.mjs:6-57` | `evals/dataset.jsonl` | 12 条客服 FAQ 的**题面风格**（短、具体、有确定答案） | 30 条三类 + `relevant_chunk_ids` + `expected_strategy` | 写入 LangSmith、无 chunk 标注、非幂等追加（D12） |
| `eval/evaluators.mjs:18-62` | `evals/judges.py` | 三个 judge 的**分工与入参映射**（groundedness 只要 context+answer；relevance 要 inputs+answer；retrieval_relevance 要 inputs+context） | `createLLMAsJudge` → 自写 40 行：结构化输出 + 重试 1 次 + `null` + `unsupported_claims` | `openevals` 依赖、`continuous` 的隐式行为、拼错的 feedbackKey |
| `eval/run_eval.mjs:13-29` | `evals/run.py` | 「被测函数 → 评测框架」的适配层只需几行（说明评测可完全解耦） | `evaluate()` → 本地 30 条循环 + 两份报告 + 门禁 + `--baseline` | LangSmith client、`result` 算完不用的写法（D13） |
| 全仓库 5 处 | `llm.py` | —— | 五处重复的客户端构造 → 单一 `Deps.llm`，重试/超时/token 统计集中在此 | 模块级客户端实例、`retrieveRelevantContent` 的 4 份副本 |

## 7. 新增项（源仓库没有、必须自己写）

| 新增 | 源实现为什么没有 | 为什么必须有 |
|---|---|---|
| numpy 精确检索 + npz | 直接把 Milvus 当默认前提（4 个文件各有一次连接代码） | 评测要离线可复现、要 `pip install` 后一条命令跑通。ANN 在百级语料下只有近似误差，没有速度收益 |
| BM25 关键词侧 | 只在 hybrid 里用 ES 实现，依赖 ik 分词插件 | 编号类查询（`ERR-4291`、`sk_live_`、`P1`）在 embedding 空间里不可分，这是向量侧的结构性弱点。无此路则「混合」不成立 |
| RRF 融合 | 源 `merge` 只 concat + 按 id 去重（D08） | 两路分数量纲不可比；用名次融合可不调参、可复现、可解释（ADR-007） |
| 带 chunk 标注的评测集 | 数据集只有 `{question, answer}`；切块无 `chunk_id` | 没有 `relevant_chunk_ids`，`context_recall` 无从计算——这是「可量化」的第一块砖 |
| 阈值门禁 + `exit 1` | 只有 LangSmith 上的可视化分数，无人判 | 分数本身不构成约束。门禁把「退步」变成 CI 里的红灯，并且指出差在哪个指标 |
| `--baseline` diff | 对比依赖 LangSmith 的实验列表 UI | 本地化后必须自己算 diff；否则「改前/改后」只能靠肉眼翻两份 JSON |
| 5 个回归测试（假模型） | 源仓库无任何测试 | 源仓库的 5 个真实缺陷（D01/D02/D03/D04/D05）**全部**是「跑一次看不出来」的类型，只有断言能抓住。且评测要花 token 和几分钟，测试必须秒级、离线 |
| `routing_accuracy` 指标 | 源仓库无「路由正确率」概念 | 路由判错时 `context_recall` 记 0 会与「召回差」混淆。单独一个确定性指标才能定位是路由问题还是检索问题（ADR-004 的连带裁定） |
| 拒答标记协议 | 源实现只说「如实告知用户」，无格式约束 | 让拒答可被规则判定，成本为零且不抖动（ADR-005） |
| 配置级消融开关 + `--ablation` | 源实现每一版都是一份独立脚本，差距靠人肉对比 | 三个布尔开关跑同一套评测，output 是「每一步 +X.Xpt」的 diff 表。这是简历里唯一可信的「效果好」证据（ADR-012） |

## 8. 弃用项与指标含义变化

| 弃用 | 替代 | 指标含义有没有变 |
|---|---|---|
| Milvus（HNSW / L2） | `numpy` 精确余弦 | `context_recall` 定义不变（都是集合覆盖），但候选来源从「ANN 近似」变成「精确 top-k」：同一 query 的结果**逐位可复现**。1,357 片段下精确检索仍是毫秒级，两者召回差异可忽略；十万级以上才需要 ANN |
| Elasticsearch（ik_max_word + 字段加权 `^2`） | `rank_bm25` + 英文词级分词 | **BM25 分数不可与他人的 ES 分数横比**（实现与分词不同）。因为 RRF 只用名次，融合结果不受影响；受影响的只有「关键词侧召回率」本身——没有词形还原与同义词扩展（`retrieve` 与 `retrieval` 是两个 token），长词召回会变差。这是 ADR-006 修订后认下的代价 |
| LangSmith 云评测 | 本地 `reports/*.json` + `*.md` | 四个指标定义不变，但失去两样东西：① 跨实验自动对比 UI → 改用 `--baseline` 手动传；② 每条样本的可视化 trace → 只有 JSON 里的 trace 摘要。**可追溯性从「平台保证」降级为「文件时间戳保证」** |
| DashScope `qwen3-rerank` | `rerank.py` 接口，默认 `enabled: false` | **这是唯一会直接压低指标的弃用**。关闭时 `selected = fused[:top_n]`，`context_recall` 与 `groundedness` 预计低于开启态。这一点必须诚实：第一版基线就是**关闭态**，开启 rerank 后的 diff 正是要证明它值不值 |
| `openevals` 的调优 prompt | 自写判官 prompt | 判官质量可能下降 → `null_rate` 上升、分数区分度变弱。缓解：强制结构化输出 `{score, reason, unsupported_claims}`，并在 README 写明判官 prompt 未经调优 |
| LangChain（`@langchain/*`） | 直接 `openai` SDK + 手写 prompt 拼装 | 无指标影响，但失去 `Document`/`Retriever` 抽象——换来「每个字段的流动都看得见」，这是 D02 那类字段名错位能被 pydantic 拦住的唯一办法 |

## 9. 风险与已知限制（三条，必须写进 README）

1. **拆解质量是上限**。`plan` 只做顺序推进（ADR-008），子问题拆错就救不回来。评测里多跳题若集中在 0.5 左右，第一嫌疑是 decompose 的 prompt，不是检索。
2. **判官不可全信**。自写判官未经调优，`groundedness` 的绝对值只能用于**同项目跨版本对比**，不能与论文/他人数字比较。`null_rate` 是它的可靠性代理指标。
3. **语料是他人论文，抽取有损**。833k 字来自 14 篇 PDF/HTML，公式与表格已做处理（删 LaTeX annotation、表格按行保留），但仍有排版噪声；也因此「语料里没有」只能用 `grep` 在**这份文本**上证明，不能声称原论文没有。扩展点：换 PDF 结构化解析（如 Marker）可提升表格质量，代价是多一个重依赖。
