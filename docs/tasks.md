# Tasks · 执行顺序

**纪律**：一次只做一条，做完停下报结果（改了什么、多少行、跑了什么命令、输出是什么），得到确认再进下一条。不允许一口气写 1000 行交付。

**每条都有「完成判据」**——判据没满足就是没完成，不许往下走。

| # | 任务 | 完成判据 | 预计行数 |
|---|---|---|---:|
| **T01** | 项目骨架：`pyproject.toml`、`.env.example`、`config.yaml`、`config.py`（yaml + env 读取 + 校验 + 6 行 `.env` 解析） | `python -c "from app.config import load_config; print(load_config())"` 打印出全部配置项；故意删一个必填环境变量时报 `ConfigError` 且提示变量名 | 40 |
| **T02** | `schemas.py`：`State`、`RouteOut`、`SubQuestionsOut`、`PlanOut`、`Chunk`、`Candidate`、`RunResult` | `python -c` 构造一个 `State` 并 `model_dump()`；把 `retrievedDocs` 这种未声明字段传进去报 `ValidationError`（**这是 D02 的防线**） | 70 |
| **T03** | `llm.py`：客户端构造、重试 2 次（指数退避）、超时、`complete` / `complete_structured`、token 累计；定义假模型的注入点（`LlmProto` 协议） | 用一个返回固定 JSON 的假对象替换客户端，`complete_structured` 能解析成 pydantic；再让它返回坏 JSON，验证重试 1 次后抛 `LLMError` 而非 `json.JSONDecodeError` | 90 |
| **T04** | 语料：`scripts/fetch_corpus.py` + 14 篇顶会论文 + `corpus.lock.json` + `NOTICE.md`；`index.py` 的切块函数 | 14 个 `.txt` 全部抽到 `## References` 节（证明正文完整）；`corpus.lock.json` 里 14 条 sha256，重抓后哈希不变；切块得 **1,357** 个 chunk（`800/100`），每个带 `chunk_id / source / heading / text`，`len(set(chunk_id)) == len(chunks)` | 脚本 260 + 切块 40 |
| **T05** | `index.py` 向量侧：批量 embedding、`.npz` 落盘、`manifest.json`（mtime + sha1）、缺失或失效自动重建 | 删除 `.index/` 后跑一次自动重建；再跑一次**不再调 embedding**（用计数器断言调用次数为 0）；改一个语料文件后 manifest 校验触发重建 | 40 |
| **T06** | `index.py` 关键词侧：小写 + 去标点 + 空白切分的 tokenizer（保留数字与 `HotpotQA` 这类词形）+ `BM25Okapi` + pickle 落盘 | `bm25("4,409 question-answer pairs")` 首条命中 CRAG 摘要（全文命中 4 次）；`bm25("IsUse")` 首条落在 `self-rag`（命中 24 次）；`bm25("HotpotQA")` 覆盖 5 篇（跨文档共享实体，也是干扰源） | 30 |
| **T07** | `retrieve.py`：入口接受**一组 query**（既有实现 hybrid 的形态），向量 + bm25 双路、`asyncio.gather` 并行、按 `chunk_id` 去重保留双分数、RRF 融合；`query_augment=true` 时先扩成 3 条改写（融合保留，默认关） | 构造两路结果（id 有交集），验证：交集文档的 RRF 分高于任一单路第一名之外的所有文档；同 id 只出现一次且带两个分数；`query_augment` 关闭时只发 1 条 query，开启时发 4 条 | 85 |
| **T08** | `rerank.py`：`enabled=false` 直通 `fused[:top_n]`；`enabled=true` 走 HTTP（URL 从配置读） | 关闭态下 `rerank(fused)` 返回对象与入参切片**逐位相同**；开启态用一个本地假 HTTP 服务验证按 `index` 回取 | 40 |
| **T09** | `nodes.py` 的 `route` + `answer`；`graph.py` 只连 simple 直边 | `--ask "1+1=?"` 端到端跑通，`trace` 只有 route/answer，`selected` 为空；模型调用计数 == 2 | 60 |
| **T10** | `nodes.py` 的 `decompose` / `retrieve` / `plan`；`graph.py` 补三条条件边（含回边） | `--ask` 一个多跳题，`--verbose` 能看到 `retrieve` 出现 ≥2 次；把 `max_rounds` 改成 1 后只出现 1 次 | 140 |
| **T11** | `cli.py`：`--ask` / `--verbose` / `--reindex`；三类错误 → 可读提示 + 非零退出码 | 断开模型 API（错 key）时退出码 1 且 stderr 含「模型」；索引目录只读时报「检索」类错误 | 60 |
| **T12** | `evals/dataset.jsonl` 30 条（14 单跳 / 10 多跳 / 6 无答案），英文，每条含 `relevant_chunk_ids`、`expected_strategy`；无答案题另附 `absence_proof` | 校验脚本：条数 30、配比正确、所有 `relevant_chunk_ids` 都在 `chunks.jsonl` 里、多跳题的 id **跨 ≥2 个 source**、参考答案里的数字能在标注片段中逐字命中、`absence_proof` 的 grep 当场重跑仍为 0 命中 | 数据 |
| **T13** | `evals/judges.py`（3 个判官 + 结构化 + 重试 1 次 + `null`）、`run.py`（循环 + 门禁 + `--limit` / `--baseline` / `--ablation`）、`report.py`（json + md + diff 表 + 消融表） | `--limit 3` 跑通并产出两份报告；把某阈值调到 1.0 后 `--gate` 退出码 1 且打印差多少；`--baseline` 输出四列表；`--ablation --limit 5` 输出四行「单路 / +BM25 / +RRF / +多跳」的 diff 表 | 330 |
| **T14** | 5 个回归测试 + 2 个补充（`test_merge_fusion` / `test_ingest_config`），全部用假模型 | `pytest -q` 7 绿，**断网可跑**，耗时 < 5 秒；逐个验证「把 bug 塞回去测试会红」（见下表） | 250 |
| **T15** | `README.md` + 真实跑一遍验收三条命令 | `pytest -q` 绿；`--ask` 有节点耗时；`evals.run --gate` 退出码 0；贴出 `wc -l` 与四个指标真实值；**署名按 ADR-013 写「融合重构」，不写「独立重写」** | 文档 |

## 进度（2026-11-26 11:20）

| # | 状态 | 实测证据 |
|---|---|---|
| T01 | 完成 | 6 条配置失败路径全部 `ConfigError` |
| T02 | 完成 | `extra="forbid"` 拦住 `retrievedDocs`；三层候选可区分 |
| T03 | 完成 | 假客户端 5 条路径；坏 JSON 抛 `LlmError` 而非 `JSONDecodeError` |
| T04 | 完成 | 14 篇 833,362 字，sha256 落锁；切块 1,357 片段，id 唯一 |
| T05 | 完成 | 建索引 1357×256；二次运行 embed 调用 0 次；改一字节即失效 |
| T06 | 完成 | `4,409…` → crag#000；`IsUse` → self-rag；`HotpotQA` → ircot/adaptive-rag |
| T07 | 完成 | RRF：双路命中 0.0325 > 单路命中 0.0159；同 id 只一次且双分保留 |
| T08 | 完成 | 关闭态逐位相同；非法下标 → `RerankError`；超时 → `RerankError(cause=TimeoutError)` |
| T09 | 完成 | `simple` 时 retrieve 调用 0 次，trace = route→answer |
| T10 | 完成 | `max_rounds=2` 时 retrieve 恰好 2 次；空召回生成调用 0 次 |
| T11 | 完成 | `--reindex` 退出码 0；占位 key 下 `--ask` 退出码 3，无 traceback |
| T12 | 完成 | 30 条（14/10/6）；`evals.validate` 七项全过；41 个引用全部存在；多跳均跨 2 个 source；6 条 absence_proof 重跑 0 命中；**负向验证 5 类错误全部报红** |
| T13 | 完成 | `judges/run/report` 落地；假模型冒烟：routing 12/14、simple 题 recall 为 None、四组消融开关两两不同、hash 报告带警告。收尾补齐三处：**消融报告真正落盘**（原先只打印，文档却指向 `reports/{ts}-ablation.md`）、基线只取普通评测（跳过 `-ablation`）、门禁失败换算成「还差几个样本」 |
| T14 | 完成 | `pytest -q` **37 个断言全绿**；6 个缺陷回填全部变红（含新增的基线过滤、样本换算）；**抓出 simple 分支误拒答的真 bug** |
| T15 | 完成 | README 八节；真实指标与阈值留空待 key（口径与「为什么留空」写在 §4）；行数：app 代码 750 / 物理 1187 |

### T12 的两处口径修正（记录在此，避免以后当成"漏标"）

- `s01` 补 `crag#000`：摘要在切块后单独成块，同样含 4,409 与五个 domain，漏标会造成系统性假阴性。
- `m10` 补 `crag#037`：与 `crag#036` 是同一段被 800 字切开的续块。
- 其余题只标「最直接的一片段」。同一事实在正文散落重复时不重复标注，因此 `context_recall` 绝对值天然偏低（ADR-018）。

## 缺陷 → 测试对齐（保证每条勘察结论都有回归保护）

| 缺陷 | 测试 | 验证「塞回去会红」的方式 |
|---|---|---|
| D01 `afterPlan` 读错字段 | `test_loop_guard` | 把条件边改成读 `strategy`，断言「retrieve 调用 2 次」立刻红 |
| D02 字段名不一致 | `test_routing` + T02 的 schema 校验 | 把节点返回键改成 `retrieveDocs`，pydantic 校验直接抛 |
| D03 真值取反 | `test_empty_retrieval` | 把 `if not candidates` 改成 `if candidates`，拒答分支断言红 |
| D04 `catch` 引用未定义变量 | `test_error_path` | 把异常包装改成引用未定义名，`NameError` 而不是 `RetrievalError` |
| D05 函数取 `.length` | `test_empty_retrieval` | 把 `if not fused` 改成 `if not fuse`（函数名），空结果分支失效，断言红 |
| D06 空召回仍生成 | `test_empty_retrieval` | 去掉 answer 的空判断，生成调用计数从 0 变 1 |
| D07 去重键兜底折叠 | `test_merge_fusion` | 把 key 改成 `supplier or "unknown"`，3 条变 1 条 |
| D08 拼接非融合 | `test_merge_fusion` | 把 RRF 换成 `dict.setdefault`，排序断言红 |
| D09 度量不一致 | `test_ingest_config` | 校验 index 与 query 同度量；故意改一处即红 |
| D10 `??` 位置 | `test_error_path` | 删掉 `MILVUS_URI` 式兜底，断言默认值不生效 |
| D11 收尾写法 | `test_error_path` | 用 `.then(err => ...)` 形态，退出码断言红 |
| D12 非幂等 | `test_ingest_config`（manifest 幂等） | 连跑两次构建，chunk 数翻倍即红 |
| D13 无产物无门禁 | T13 完成判据 | 不写报告时 `--gate` 找不到基线 |
| D14 `json` 先于 `ok` | `test_error_path` | 假服务返回 HTML 500，断言 message 含 `500` |
| D15 无降级 | `test_error_path` | 让向量侧抛错，断言单路仍能答（`retrieval_degraded`） |

## 五个必测用例的具体断言（写代码前先定死）

1. `test_routing.py` — `simple` 不进 `retrieve`（`retrieve` 计数 0）；`complex` 必进 `decompose`（`decompose` 计数 1）。
2. `test_loop_guard.py` — 规划恒返回 `retrieve`，`max_rounds=2` → `retrieve` 计数 `== 2`；`trace` 末尾是 answer。
3. `test_empty_retrieval.py` — 召回全空 → 生成调用 `== 0`，`answer` 以 `I cannot confirm this from the provided sources` 开头。
4. `test_error_path.py` — 向量侧抛网络异常 → 单路继续（`retrieval_degraded` 在 trace 里）；两路都抛 → `RetrievalError`，退出码非 0，无未捕获异常。
5. `test_happy_path.py` — 单跳题端到端：`answer` 非空、`selected` 非空、`trace` 顺序为 `route → decompose → retrieve → plan → answer`。

## 停止点

- T04 之后停一次：语料是他人论文，抽取质量与「哪些事实可被问」由你过一眼——我会给你 14 篇的字数/结构表、3 条真实抽取片段，以及 6 条无答案题的 `absence_proof`。
- T10 之后停一次：多跳链路是核心，先看真实 `--verbose` 输出确认「图真的回到了 retrieve」。
- T12 之后停一次：30 条题面逐条过一遍，尤其是多跳题是否真的跨了两篇——这是最容易被我自己骗过去的地方。
