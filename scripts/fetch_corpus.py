"""抓取评测语料：14 篇 RAG 方向顶会论文的 arXiv 定版全文，转成纯文本。

为什么用脚本而不是把全文提交进仓库
--------------------------------
arXiv 默认许可是「arXiv.org perpetual, non-exclusive license」，授权的是阅读与
引用，不含再分发。仓库里只锁「抓哪一版 + 抓下来的 sha256」，全文进 .gitignore。
任何人 clone 后执行：

    python scripts/fetch_corpus.py

拿到逐字节相同的文本 → chunk_id 稳定 → 评测结果可复现。这比拷文件更可靠：
拷进仓库的全文一旦被手工改过，没人知道基线是拿哪份文本跑出来的。

用法
----
    python scripts/fetch_corpus.py            # 缺哪篇抓哪篇，哈希不符则重抓
    python scripts/fetch_corpus.py --force    # 全部重抓
    python scripts/fetch_corpus.py --only crag
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
LOCK_PATH = ROOT / "data" / "corpus.lock.json"
UA = {"User-Agent": "agentic-rag-eval/1.0 (academic corpus fetch)"}

# 选文原则：只取 RAG 主干上的顶会定版（NeurIPS / ICLR / ACL / EMNLP / NAACL / TACL），
# 且相互之间有真实引用关系——多跳题靠这些引用链成立，不靠剧本。
PAPERS = [
    dict(slug="rag-lewis", id="2005.11401", ver=4,
         title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
         venue="NeurIPS 2020", ar5iv=True),
    dict(slug="beir", id="2104.08663", ver=4,
         title="BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
         venue="NeurIPS 2021 Datasets & Benchmarks", ar5iv=True),
    dict(slug="react", id="2210.03629", ver=3,
         title="ReAct: Synergizing Reasoning and Acting in Language Models",
         venue="ICLR 2023", ar5iv=True),
    dict(slug="ircot", id="2212.10509", ver=2,
         title="Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions",
         venue="ACL 2023", ar5iv=True),
    dict(slug="hyde", id="2212.10496", ver=1,
         title="Precise Zero-Shot Dense Retrieval without Relevance Labels",
         venue="ACL 2023", ar5iv=True),
    dict(slug="rankgpt", id="2304.09542", ver=3,
         title="Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents",
         venue="EMNLP 2023", ar5iv=True),
    dict(slug="flare", id="2305.06983", ver=2,
         title="Active Retrieval Augmented Generation",
         venue="EMNLP 2023", ar5iv=True),
    dict(slug="lost-in-the-middle", id="2307.03172", ver=3,
         title="Lost in the Middle: How Language Models Use Long Contexts",
         venue="TACL 2024", ar5iv=True),
    dict(slug="ragas", id="2309.15217", ver=2,
         title="Ragas: Automated Evaluation of Retrieval Augmented Generation",
         venue="EACL 2024 Demo", ar5iv=True),
    dict(slug="self-rag", id="2310.11511", ver=1,
         title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
         venue="ICLR 2024", ar5iv=True),
    dict(slug="ares", id="2311.09476", ver=2,
         title="ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems",
         venue="NAACL 2024", ar5iv=True),
    dict(slug="adaptive-rag", id="2403.14403", ver=2,
         title="Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity",
         venue="NAACL 2024", ar5iv=False),
    dict(slug="crag", id="2406.04744", ver=2,
         title="CRAG -- Comprehensive RAG Benchmark", venue="NeurIPS 2024 D&B",
         ar5iv=False),
    dict(slug="ragchecker", id="2408.08067", ver=2,
         title="RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation",
         venue="NeurIPS 2024", ar5iv=False),
]


def _get(url: str, tries: int = 3) -> bytes | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception as e:  # 404 换下一个源，其它错误退避重试
            print(f"    ! {type(e).__name__}: {url} (try {i + 1}/{tries})", file=sys.stderr)
            time.sleep(2 * (i + 1))
    return None


def _clean(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u2212", "-")
    return re.sub(r"\s+", " ", s).strip()


def html_to_text(html: str) -> str:
    """ar5iv / arXiv HTML 版 → 带 `## 标题` 的纯文本，表格按行保留。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    # <annotation> 存的是同一段公式的 LaTeX，留着会让每个数字出现两遍（"63 % 63\%"），
    # 直接把 BM25 的 term 频次带偏。
    for bad in soup.find_all(["script", "style", "nav", "footer", "annotation"]):
        bad.decompose()
    main = None
    for sel in ("article.ltx_document", "div.ltx_page_main", "main", "body"):
        main = soup.select_one(sel)
        if main:
            break
    if main is None:
        return ""

    out: list[str] = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "figcaption"]):
        # 表格内部嵌套的 <p> 由表格自己输出，避免重复
        if el.name in ("p", "li") and el.find_parent("table") is not None:
            continue
        # 参考文献条目不进语料（噪声大、对回答问题无帮助），但正文列表要进
        if el.name == "li" and el.find_parent(class_=re.compile("biblist|ltx_bibliography")):
            continue
        if el.name == "figcaption":
            fig = el.find_parent("figure")
            if fig is not None and fig.find("table") is not None:
                continue  # 表题由表格分支处理
            cap = _clean(el.get_text(" ", strip=True))
            if len(cap) >= 15:
                out.append("[CAP] " + re.sub(r"^(Figure|Fig\.)\s*\d+[:.]\s*", "", cap))
            continue
        if el.name == "table":
            fig = el.find_parent("figure")
            cap = (fig.find("figcaption") if fig else None) or el.find("figcaption")
            if cap:
                c = _clean(cap.get_text(" ", strip=True))
                c = re.sub(r"^Table\s*\d+[:.]\s*", "", c)
                if c:
                    out.append("")
                    out.append(f"[TABLE] {c}")
            for tr in el.find_all("tr"):
                cells = [_clean(c.get_text(" ", strip=True)) for c in tr.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if cells:
                    out.append(" | ".join(cells))
            continue
        txt = _clean(el.get_text(" ", strip=True))
        if el.name.startswith("h"):
            if len(txt) < 3:  # 空标题保留会污染结构，但 "3 Method" 这类必须留
                continue
            out.append("")
            out.append("## " + txt)
        else:
            if len(txt) < 20:  # 纯公式/编号碎片，进语料只会污染 BM25
                continue
            out.append(txt)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def pdf_to_text(pdf: bytes) -> str:
    """兜底：HTML 版拿不到时用 pypdf 抽，质量差但仍可用。"""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf))
    pages = [_clean(p.extract_text() or "") for p in reader.pages]
    return "\n\n".join(p for p in pages if p)


def build_one(p: dict, force: bool) -> dict | None:
    slug = p["slug"]
    dst = CORPUS_DIR / f"{slug}.txt"
    if dst.exists() and not force:
        print(f"  = {slug:18s} 已存在，跳过")
        return None

    text = ""
    used = ""
    if p["ar5iv"]:
        urls = [f"https://arxiv.org/html/{p['id']}v{p['ver']}",
                f"https://ar5iv.labs.arxiv.org/html/{p['id']}v{p['ver']}"]
    else:
        urls = [f"https://arxiv.org/html/{p['id']}v{p['ver']}",
                f"https://arxiv.org/pdf/{p['id']}v{p['ver']}"]

    for url in urls:
        raw = _get(url)
        if not raw:
            continue
        if url.endswith(".pdf") or raw[:4] == b"%PDF":
            text = pdf_to_text(raw)
        else:
            text = html_to_text(raw.decode("utf-8", "ignore"))
        if len(text) > 8000:
            used = url
            break
        text = ""

    if not text:
        print(f"  ✗ {slug:18s} 所有源都失败", file=sys.stderr)
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text + "\n", encoding="utf-8")
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    print(f"  ✓ {slug:18s} {len(text):>7d} 字  {len(text.split()):>6d} 词  sha256={digest[:12]}")
    return dict(slug=slug, arxiv_id=p["id"], version=p["ver"], url=used,
                sha256=digest, chars=len(text), venue=p["venue"],
                citation=f"{p['title']}. {p['venue']}. arXiv:{p['id']}v{p['ver']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已有文件也重抓")
    ap.add_argument("--only", default=None, help="只抓某个 slug")
    args = ap.parse_args()

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8")) if LOCK_PATH.exists() else {"papers": []}
    by_slug = {e["slug"]: e for e in lock["papers"]}

    papers = [p for p in PAPERS if args.only is None or p["slug"] == args.only]
    print(f"目标 {len(papers)} 篇 → {CORPUS_DIR}")
    ok = 0
    for p in papers:
        entry = build_one(p, args.force)
        if entry:
            by_slug[entry["slug"]] = entry
            ok += 1

    lock["papers"] = sorted(by_slug.values(), key=lambda e: e["arxiv_id"])
    lock["note"] = ("全文不提交仓库（arXiv 默认许可不含再分发权）。"
                    "执行 scripts/fetch_corpus.py 后本地文本的 sha256 必须与本文件一致，"
                    "否则 chunk_id 与评测基线全部失效。")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    total = sum(e["chars"] for e in lock["papers"])
    print(f"\n锁定 {len(lock['papers'])} 篇，合计 {total:,} 字 → {LOCK_PATH.relative_to(ROOT)}")
    print(f"本次新增/更新 {ok} 篇")
    return 0 if len(lock["papers"]) == len(PAPERS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
