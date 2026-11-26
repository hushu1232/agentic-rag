# 既有实现勘察报告（阶段 0）

- 勘察对象：与导师共同完成的既有实现（只读勘察；原件与本地路径不写入本仓库）
- 勘察范围：13 个 `.mjs`，合计 2249 行（实测 `wc -l`，与附录逐行一致）
- 本阶段不产出任何项目代码

---

## ① 文件职责表（13 行）

| 文件 | 行数 | 它到底做了什么 | 依赖 | 复用价值 |
|---|---:|---|---|---|
| `advanced-rag/src/naive-rag.mjs` | 192 | 最小直线图 `retrieve → generate`：Milvus 相似度检索 top5，把 `chapter_num + content` 拼成片段块塞进 prompt，流式打印回答。固定问「阿朱的结局」，无入参。 | langchain/openai, langgraph, zod(未用), @langchain/community milvus | 中：`documents` 结构、片段拼装模板、Milvus HNSW 连接参数 |
| `advanced-rag/src/rag-query-router.mjs` | 238 | 前置 `route_question` 节点用 zod 结构化输出把问题分 `simple/complex`；`decideNext` 读 `state.strategy` 走 `direct_answer` 或 `retrieve`；simple 直答不检索。 | 同上 | 高：**路由条件边的原始形态**、结构化输出写法 |
| `advanced-rag/src/rag-multihop.mjs` | 410 | 在 router 上加两段：`decompose_question` 用 zod 拆 1–8 条**有序**子问题（禁指代词、须可独立检索）；`plan_next_step` 输出 `{nextAction, reason}` 并带两道代码层硬熔断（轮数、剩余子问题），`retrieve` 按 `nextSubIdx` 推进、`mergeUnique` 按 id 去重保高分。 | 同上 | 最高：**多跳循环 + 熔断 + 去重**，本项目 §3 状态机的直接来源 |
| `advanced-rag/src/rag-webfallback.mjs` | 378 | 本地检索后加 `evaluate_local` 用 zod 判 `{enough, missing, reason, web_query}`；不够则 `web_search`（博查 API）拿网页摘要写进 `webContext`，再回 `evaluate_local` 二次判定，最后 `generate` 融合两路上下文。 | 同上 + HTTP fetch + `BOCHA_API_KEY` | 高：**充分性评估 + 兜底回边 + 二次评估短路条件**的形态 |
| `advanced-rag/src/rag/hybrid-retrieval.mjs` | 265 | 独立编译的混合召回子图：`query_augment`（3 条改写）→ 并行 `es_recall`（`multi_match` + `ik_smart`，标题加权 ^2）与 `milvus_recall` → `merge`（concat + 按 id 去重）→ `rerank`（DashScope `qwen3-rerank` topN=3）→ `generate_answer`（有/无上下文两套 prompt）。 | @elastic/elasticsearch, milvus, langchain prompts, 自写 rerank 类 | 高：**双路并行召回 + 并行汇合边**的写法；但融合逻辑本身要重写（见 D08） |
| `advanced-rag/src/rag/query-augment.mjs` | 48 | 一个 `withStructuredOutput` 让 LLM 产出**恰好 3 条**不同角度的检索问句；`normalizeThreeQueries` 用原句补齐到 3 条；`retrievalQueryString` 把原句 + 3 条拼成召回用 query 列表。 | @langchain/core prompts, zod | 中：**查询扩展**这一步的接口形状（`augment_query` 可直接对应到配置项） |
| `advanced-rag/src/rag/seed-data.mjs` | 279 | 10 条「生活笔记」种子数据，双写：ES 建 `ik_max_word` 索引 + bulk 写入；Milvus 建 collection（HNSW/L2）、embed 后 insert、flushSync。 | ES client, milvus2-sdk-node, langchain openai | 中：**双库同源写入 + 字段对齐**这件事的参照（也暴露了 D09/D16 的不对齐） |
| `advanced-rag/src/rerank/dashscope-rerank.mjs` | 48 | 继承 `BaseDocumentCompressor`，POST `qwen3-rerank`，按 `output.results[].index` 从原 docs 里取回重排后的 topN 条。 | @langchain/core document_compressors | 中：**重排接口的可替换外壳**（换成任意 rerank HTTP 服务只需改 body 映射） |
| `langsmith-demo/src/rag_agent.mjs` | 85 | 极简 `retrieve → generate` 图（`asRetriever(k=4)`），导出 `ragApp` 与 `ask(question) → {answer, context}`，供评测层调用。 | langgraph, langchain prompts/output_parsers, milvus | 高：**「被测系统」暴露给评测的接口**——`ask()` 返回 `answer + context` 正是算 `context_recall / groundedness` 需要的两个入参 |
| `langsmith-demo/src/eval/build_dataset.mjs` | 86 | 12 条客服 FAQ 的 `{inputs.question, outputs.answer}` 硬编码数组，`readDataset` 不存在就 `createDataset`，然后 `createExamples` 写入 LangSmith 云。 | langsmith SDK | 中：**数据集字段形状**可参考；但只有 `answer` 没有 `relevant_chunk_ids`（见 D13），且写到云端要整个换掉 |
| `langsmith-demo/src/eval/evaluators.mjs` | 67 | 用 `openevals` 的 `createLLMAsJudge` 造 3 个 judge：groundedness、helpfulness、retrieval_relevance，`continuous: true` 出 0–1 分；再包成 `({inputs, outputs}) => judge(...)` 的 evaluator 数组。 | openevals, langchain openai | 高：**三个 judge 的分工与入参映射**（groundedness 只需 context+answer，relevance 只需 inputs+context） |
| `langsmith-demo/src/eval/run_eval.mjs` | 34 | 把 `ask()` 包成 `runRagAgent`（context 降维成 `pageContent[]`），交给 LangSmith `evaluate()`，`maxConcurrency: 2`。 | langsmith evaluation | 中：**「被测函数 → 评测框架」的适配层**只需 7 行，说明评测层可以完全解耦 |
| `langsmith-demo/src/milvus_insert.mjs` | 119 | 读 `./data/*.{text,md}`，`RecursiveCharacterTextSplitter(500/50)` 切块，建 collection（IVF_FLAT/L2）+ 索引 + load + insert。 | fs/path, milvus2-sdk-node, textsplitters, openai | 中：**切块参数与 metadata 形状**（只有 `source`，没有 `chunk_id` → 见 D13） |

> 附注：`retrieveRelevantContent` 在本次范围内被复制了 **4 份**——`naive-rag:39`、`rag-query-router:98`、`rag-multihop:113`、`rag-webfallback:47`，函数体逐字相同（含两行被注释掉的 `// doc.pageContent`）。第 5 份在范围外的 `ai/agent_in_action/tlbb/src/rag.mjs:48`。这 4 份必须在新项目里合并成单一 `retrieve.py`，否则会被抄第 5 份。

---

## ② 演进关系（5 个 `rag-*.mjs` 是同一件事的 5 个递进版本）

```
naive            START → retrieve ──────────────→ generate → END
                          （所有问题都检索，单查询，top5）

router           START → route ─┬ simple ────────→ direct_answer → END
                                └ complex ──→ retrieve → generate → END

multihop         START → route ─┬ simple ────────→ direct_answer → END
                                └ complex → decompose → retrieve → plan ─┐
                                              ↑                          │
                                              └──── retrieve ────────────┤
                                                                         └→ generate → END
                                          （plan 写 plannedNext，边却读 strategy ← D01）

webfallback      START → route ─┬ simple ────────→ direct_answer → END
                                └ complex → local_retrieve → evaluate_local ─┐
                                                    ↑                        │
                                                    └──── web_search ←───────┤
                                                                             ├ enough → generate → END
                                                                             └ !enough → web_search

hybrid           START → query_augment ─┬→ es_recall ────┐
                                        └→ milvus_recall ┴→ merge → rerank → generate → END
```

| 版本 | 解决了上一版的什么缺陷 | 又留下了什么缺陷 |
|---|---|---|
| **naive** | ——（基线：证明「检索 + 拼上下文 + 生成」这条路能跑通） | 不判断问题类型，`1+1=?` 也去检索小说库；单查询单跳；**召回为空照样生成**（D06），等于给模型一个空上下文让它自由发挥 |
| **router** | 加了「要不要检索」这一层判断：常识题不再被检索结果污染，省掉一次无意义的向量查询 | complex 分支仍只检索一次。像「四大恶人之二的儿子，生父公开身份是什么」这种需要先定实体、再查属性的链式问题，一次 top5 召回只能拿到链的一端；检索只按原问题，不做改写 |
| **multihop** | 把 complex 问题拆成**有序子问题**再逐个检索，用 `mergeUnique` 跨轮去重（避免重复片段让模型误以为"多方证据"），并在代码层加了两道熔断防死循环 | ① 熔断写对了，但**条件边读错字段**（D01）：`afterPlan` 读 `state.strategy`（值恒为 `"complex"`），永远不等于 `"retrieve"`，所以图从不回到 `retrieve`——多跳退化成单跳，且不报错。② 是否继续检索只看「剩余子问题条数」，没有「召回质量」信号：第一条子问题就召回了答案也会继续跑完。③ `mergeUnique` 用 `String(d.id)` 当 key，而 `id` 兜底是 `"unknown"`（D07），无 id 文档全被折成 1 条 |
| **webfallback** | 补上「知识库里根本没有」的场景（2013 版电视剧集数）：LLM 判 `enough=false` 就走联网，二次评估后短路到生成 | ① 本地召回结果写进了 `retrieveDocs`，但状态声明的是 `retrievedDocs`（D02）——`state.retrievedDocs` 永远是 `[]`。② 网页结果判空**真值写反**（D03）：有结果时返回「未找到相关结果。」。③ 兜底没有轮数上限，`webContext` 为空就会 `evaluate → web_search → evaluate` 空转（当前被 D03 掩盖，修好 D03 就暴露）。④ `evaluate` 只判「上下文够不够」，不判「召回的内容对不对」——召回一堆不相干的片段，`enough` 也可能判 true |
| **hybrid** | 换掉「单路向量」这个根问题：专有名词与编号（`PO-20250409-K9`、`SN-MILO-77821`、`Q7` 型号）在 embedding 空间里几乎不可分，BM25 的精确匹配正好补上；3 条改写查询并行召回扩大覆盖；rerank 用 cross-encoder 对候选做精排压掉长尾噪声 | ① `merge` 只是 concat + 按 id 去重，**ES 先入为主**，Milvus 独有文档的分数直接丢失（D08），没有 RRF/加权，两路召回量纲不同的好处没有被利用。② `if (!merge.length)` 把函数当数组取 `.length`（D05）→ 空结果短路失效，空数组直送 rerank API。③ 整条链无降级：ES / Milvus / rerank 任一失败，`graph.invoke` 直接抛原始网络错误（D15） |

**为什么最后落在「混合检索 + 重排」**：前面四版都在**编排层**加分支（要不要检索 / 拆不拆 / 要不要联网），但召回本身一直是「一条 query → 一个向量库 → top5」。编排再聪明，召回不到就是召回不到。hybrid 第一次动了召回层：把「语义相似」（向量）和「字面命中」（BM25）两路独立召回，再用 rerank 在**候选池**上做一次 query-doc 联合打分。代价是候选多了、分数不可比了——所以既有实现在 `merge` 这一步就崩了（D08），只做了去重没做融合。这正是新项目必须自己补 RRF 的地方，也是「混合检索 + 重排」在既有实现里只完成了 2/3 的证据。

---

## ③ 缺陷清单

逐文件核对逻辑正确性。**15 条可复现缺陷**（均有断言），另附 5 条低价值但会误导排查的问题。「落到用例」列对应阶段 2 的 5 个回归测试 + 2 个补充用例（`test_merge_fusion.py`、`test_ingest_config.py`）。

| # | 位置 | 问题（机制） | 后果 | 断言（当前实现会失败） | 落到用例 |
|---|---|---|---|---|---|
| **D01** | `rag-multihop.mjs:233-235` | `afterPlan` 读 `state.strategy`，但 `planNextStepNode` 写入的字段是 `plannedNext`（:285）。`strategy` 在 `route_question` 里被写成 `"complex"`，与 `"retrieve"` 永不相等。 | 条件边恒走 `generate`；`retrieve → plan → retrieve` 这条回边从不执行。多跳能力形同不存在，且静默失败（不报错、日志里 `plannedNext` 还打印得好好的）。 | plan 的 LLM 固定输出 `retrieve` 且 `max_rounds=2` 时，`retrieve` 节点调用次数应 `== 2`；当前 `== 1`。 | `test_loop_guard` |
| **D02** | `rag-webfallback.mjs:21` vs `:126-129` | 状态声明字段 `retrievedDocs`，`retrieveLocalNode` 返回的对象键却是 `retrieveDocs`。 | LangGraph 只合并已声明的通道，未知键被丢弃 → `state.retrievedDocs` 恒为 `[]`。任何依赖它做引用标注 / 评测 `context_recall` 的下游都是空数据；`main` 里也没有任何地方能发现这个字段一直空着。 | `invoke` 后 `result.retrievedDocs.length` 应 `> 0`（本地库有数据时）；当前 `== 0`。 | 补充：`test_error_path` 的姊妹用例 |
| **D03** | `rag-webfallback.mjs:265-268` | `if (webpages.length) { return "未找到相关结果。" }` —— 判空与返回文案的真值写反。 | 联网搜到结果时反而丢弃全部结果、回一句「未找到」；搜不到时走 `.map` 得到空串 `""`。web 兜底这条路在两种情况下都拿不到上下文。 | mock 返回 `data.webPages.value` 为 3 条时，`webContext` 应含 `标题：`；当前 `== "未找到相关结果。"`。 | `test_empty_retrieval` |
| **D04** | `rag-webfallback.mjs:245-247` | `catch { throw new Error(\`...(网络错误): ${error.message}\`) }` —— `catch` 未绑定变量却引用 `error`。 | 网络异常时抛的是 `ReferenceError: error is not defined`，原始错误信息与 cause 全部丢失；上层看到的是「error is not defined」，与真正的超时/断网无关。 | mock `fetch` reject 时抛出的 `Error.message` 应含 `网络错误`；当前抛 `ReferenceError`。 | `test_error_path` |
| **D05** | `hybrid-retrieval.mjs:219` | `if (!merge.length) return { topDocuments: [] }` —— `merge` 是同文件 :71 定义的**函数**，`.length` 是形参个数（2），`!2 === false`。 | 短路分支永不进入。合并结果为空时照样调用 `reranker.compressDocuments([], query)`，把空数组发给 rerank HTTP 接口 → 抛 HTTP/解析错误；`generate_answer` 里精心写的 `NO_CONTEXT_PROMPT` **永远到不了**。 | `merged=[]` 时 reranker 调用次数应 `== 0` 且 `answer` 走无上下文文案；当前 reranker 被调用一次并抛错。 | `test_empty_retrieval` |
| **D06** | `naive-rag.mjs:168-172`（配 `:68-73`） | 空召回判断写在 `main()` 里，且在 `graph.invoke()` **之后**——`generateNode` 早已用空 `context` 调完 LLM。 | 终端先流式打印一段模型自行编造的回答，再打印「未找到相关内容」。用户看到的是「编了，然后又说没找到」。这就是必须杜绝的「把空结果当有结果」。 | `documents=[]` 时 LLM `stream` 调用次数应 `== 0`；当前 `== 1`。 | `test_empty_retrieval` |
| **D07** | `rag-multihop.mjs:174-186`，配 `:120` 的 `id: doc.metadata?.id ?? "unknown"` | `mergeUnique` 用 `String(d.id)` 当去重 key；metadata 无 id 时全部兜底为字符串 `"unknown"`，于是所有无 id 文档共享同一个 key，只保留分数最高的一条。 | 静默丢文档：5 条无关片段只活下来 1 条，且丢弃过程没有任何日志（`merged.length` 看起来"去重生效了"，实际是数据被吞）。多跳场景下后续子问题的证据可能整段消失。 | 传入 3 条 `metadata.id` 缺失、内容不同的文档，`merged.length` 应 `== 3`；当前 `== 1`。 | `test_happy_path`（断言召回条数）+ 补充用例 |
| **D08** | `hybrid-retrieval.mjs:71-89` | `merge` = `[...esDocs, ...milvusDocs]` 后 `dedupeDocsById`，同 id **保留先到的 ES 文档**；`docFromEsHit`（:57-67）根本没写 `score` 字段。 | ① 不是融合，是「拼接 + 去重」：Milvus 的余弦相似度完全没参与排序，两路召回量纲不同的价值为零。② 同 id 文档保留 ES 版本 → 分数字段丢失，`merged` 里一半文档没有分数。③ 传给 rerank 的候选顺序 = ES 命中顺序，向量路的高置信结果可能排在候选池尾部被 topN 截掉。 | `merged` 中每条应带可比的分数字段（或已按 RRF 排序）；当前 ES 来源文档 `assert "score" in doc.metadata` 失败。 | 补充：`test_merge_fusion` |
| **D09** | `rag/seed-data.mjs:225` vs `hybrid-retrieval.mjs:24-29` | 建索引用 `MetricType.L2`（HNSW），但检索端 `Milvus.fromExistingCollection` 未传度量，LangChain 的 Milvus 默认按 `COSINE` 检索。 | 索引度量与查询度量不一致。Milvus 严格校验下直接报 `metric type not match`；宽松时返回的 distance 语义混乱（L2 距离 vs 余弦相似度），跨库分数更没法比。 | 建索引与检索的 `metric_type` 应一致；当前 `L2 != COSINE`。 | 补充：`test_ingest_config` |
| **D10** | `milvus_insert.mjs:16` | `process.env.MILVUS_URI.replace(...) ?? "localhost:19530"` —— `??` 只兜 `replace()` 的**返回值**，兜不住 `undefined.replace` 本身。 | 未设 `MILVUS_URI` 时先抛 `TypeError: Cannot read properties of undefined (reading 'replace')`，默认值永远用不上；而 `.env.example` 里若没写这一项，跑起来必然崩在这行。 | `unset MILVUS_URI` 时 `MILVUS_ADDRESS === "localhost:19530"`；当前抛 `TypeError`。 | `test_error_path`（配置缺失分支） |
| **D11** | `naive-rag.mjs:189-190`、`rag-query-router.mjs:237-238`、`rag-multihop.mjs:409-410` | 三种收尾写法全错：① `.catch(err => console.error)` 传的是函数引用，`console.error` 被当作 handler 调用（丢失 err）；② `.then(err => console.error(err))` 的形参其实是**成功值** `undefined`；③ `webfallback:328-379` 干脆 `main()` 裸调用，无 handler。 | ② 成功时打印一行 `undefined`（看起来像报错）；失败时 promise rejection **无人 catch**，Node 只会打印 warning 后继续，进程退出码仍为 0。CI / 脚本里永远判定「通过」。 | 注入一个必定抛错的节点，进程退出码应 `!= 0` 且 stderr 含原始错误；当前退出码 `0`，stdout 多一行 `undefined`。 | `test_error_path` |
| **D12** | `eval/build_dataset.mjs:64-72` | `readDataset` 成功（数据集已存在）时**不 return**，继续执行 `createExamples` 追加 12 条。 | 非幂等：跑两次评测集变 24 条，跑三次 36 条。重复样例会双重计入均值，跨次实验的指标不可比——这正好毁掉「基线 diff」这件事的前提。 | 连续执行两次后数据集条数应 `== 12`；当前 `== 24`。 | 补充：评测层自检 |
| **D13** | `eval/run_eval.mjs:21-29`、`eval/build_dataset.mjs:6-57`、`milvus_insert.mjs:40-46` | ① `const result = await evaluate(...)` 之后再无引用，无落盘、无阈值、无退出码；② 数据集只有 `{question, answer}`，**没有 `relevant_chunk_ids`**，`context_recall` 无从计算；③ 切块只写 `metadata.source`，没有稳定 `chunk_id`，即便想标注也标不了。 | 评测跑完只在 LangSmith 云端留一个 experiment，本地零产物；「改前/改后」的分数对比、门禁、失败样本 top5 全部不可能实现。这是评测层必须重写而不是移植的根本原因。 | 跑完应存在 `reports/eval-*.json` 且含 4 个指标；当前无任何本地产物。 | 评测层自检 |
| **D14** | `rerank/dashscope-rerank.mjs:34-46` | 先 `const json = await res.json()` 再检查 `res.ok`；随后 `results.map(item => documents[item.index])` 不校验下标范围。 | ① 服务端返回非 JSON 的 5xx（网关 HTML 错误页）时抛 `SyntaxError`，状态码信息全丢，排查时看不到 `500`。② `index` 越界时数组里出现 `undefined`，到 `formatDocsAsContext` 读 `d.metadata` 时抛 `TypeError`，错误点离根因很远。 | 接口返回 500 + HTML 时，错误信息应含 `500`；当前 `SyntaxError: Unexpected token '<'`。 | `test_error_path` |
| **D15** | `hybrid-retrieval.mjs:170-223`（`es_recall`/`milvus_recall`/`rerank` 三节点） | 三个外部调用节点均无 try/catch，也没有任何降级分支；`Promise.all` 任一 reject 即整链失败。 | ES 或 Milvus 抖动一次，用户的提问就得到一整个堆栈；而这条链里「只剩一路召回」是完全可用的降级态（BM25 挂了向量还能答）。编排层花了 4 个版本学「不够就兜底」，召回层却一次兜底都没有。 | 让 ES client 抛网络错，`invoke` 应返回结构化失败（`RetrievalError`）而非原始驱动异常；当前直接抛 `@elastic/elasticsearch` 的 `ConnectionError`。 | `test_error_path` |

**低价值但会误导排查的 5 条**（不单列断言，实现时一并清理）：

| # | 位置 | 问题 |
|---|---|---|
| D16 | `hybrid-retrieval.mjs:92` | 模板字符串里写成 `$(docs?.length ?? 0)`，少一个 `{`，日志永远打印字面量 `$(docs?.length ?? 0)`。 |
| D17 | `seed-data.mjs:258` | `\`Milvus 写入完成 {insert_cnt} ${cnt}\``，`{insert_cnt}` 缺 `$`，插值不生效。 |
| D18 | `rag-webfallback.mjs:368` | 初始 state 里写 `routeReson:"")`，正确字段是 `routeReason`——拼错不报错（Annotation 通道宽松），但说明整条状态无类型约束。 |
| D19 | `rag/query-augment.mjs:24-31` | 扩展失败时用**原句**补齐到 3 条，`retrievalQueryString` 得到 `[q,q,q,q]` 4 条 → 同一 query 被 ES 与 Milvus 各检索 4 次，成本 ×4、结果全重复。降级应返回 `[q]` 而不是 `[q,q,q]`。 |
| D20 | `naive:175-179`、`hybrid:195`、`seed-data:124-171`、`milvus_insert:62,72`、`webfallback:264` | 调试残留：`console.log(batches)`、`console.log(chunks)`（打印全部切块）、`console.log(vectors)`（打印 1024 维向量）、`console.log(json, "///////////")`、`"zhaugntai"` 拼音错字。 |

**关于「未发现」**：13 个文件里没有找到「空字符串 / 空数组真值判断」之外的隐藏分支错误（D03 是唯一一处真值取反）；`documents ?? []`、`state.documents.length` 这类空值保护在多处缺失，但只有 D06/D07/D15 会真的导致错答或崩溃，其余只是脆弱。没有编造问题凑数——上面 15 条的每一条我都能指出具体行号、触发条件与失败断言。

---

阶段 0 结束。请确认 `docs/source-audit.md`，确认后我再输出阶段 1 的 `docs/merge-mapping.md`（谁去哪、哪些丢掉、新增项与弃用项）。
