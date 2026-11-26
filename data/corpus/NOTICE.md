# 语料来源与许可

本目录的 `.txt` 文件由 `scripts/fetch_corpus.py` 抓取生成，**不提交进仓库**（`.gitignore`）。
仓库只提交 `../data/corpus.lock.json`：arXiv 编号、定版号、抓取 URL、`sha256`、字数。

复现：

```bash
python scripts/fetch_corpus.py          # 缺哪篇抓哪篇
python scripts/fetch_corpus.py --force  # 全部重抓
```

抓取后每个文件的 `sha256` 必须与 `corpus.lock.json` 一致。哈希不符说明抓到的不是同一版文本，
此时 chunk_id 与仓库里的评测基线全部失效——**不要带着不一致的语料跑评测**。

## 许可说明

arXiv 的默认许可是 *arXiv.org perpetual, non-exclusive license*，授予阅读与引用，
**不授予再分发权**。因此本仓库不包含任何论文全文，只包含指向它们的标识符与哈希。
各篇的具体许可（CC BY / CC BY-SA / arXiv 默认）以 arXiv 页面标注为准。

## 论文清单

| slug | 论文 | 发表处 |
|---|---|---|
| `rag-lewis` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | NeurIPS 2020 |
| `beir` | BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models | NeurIPS 2021 Datasets & Benchmarks |
| `react` | ReAct: Synergizing Reasoning and Acting in Language Models | ICLR 2023 |
| `ircot` | Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions | ACL 2023 |
| `hyde` | Precise Zero-Shot Dense Retrieval without Relevance Labels | ACL 2023 |
| `rankgpt` | Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents | EMNLP 2023 |
| `flare` | Active Retrieval Augmented Generation | EMNLP 2023 |
| `lost-in-the-middle` | Lost in the Middle: How Language Models Use Long Contexts | TACL 2024 |
| `ragas` | Ragas: Automated Evaluation of Retrieval Augmented Generation | EACL 2024 Demo |
| `self-rag` | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | ICLR 2024 |
| `ares` | ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems | NAACL 2024 |
| `adaptive-rag` | Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity | NAACL 2024 |
| `crag` | CRAG — Comprehensive RAG Benchmark | NeurIPS 2024 Datasets & Benchmarks |
| `ragchecker` | RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation | NeurIPS 2024 |

引用格式：作者. 标题. 会议/期刊, 年份. arXiv:编号 v版本.
