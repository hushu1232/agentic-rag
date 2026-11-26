"""索引层：切块、向量化、落盘、失效校验。

切块规则唯一目的是**让标注可复现**：同一份语料 + 同一组参数 → 同样的 `chunk_id`，
否则 `evals/dataset.jsonl` 里的 `relevant_chunk_ids` 会随一次重跑集体失效。所以这里
不依赖模型、不依赖 token 计数，只按字符数切；被切断的地方由 `overlap` 兜住，且拼
context 时会把 `heading` 一起带上。

向量化有两个实现，走同一个 `Embedder` 协议（ADR-015）：
- `OpenAIEmbedder`：真实语义向量，需要 `OPENAI_API_KEY` 与可用的 `base_url`。
- `HashEmbedder`：确定性词袋哈希向量。**它不是语义向量**，只保证「有词面重叠就更近」，
  用于无 key 时验证落盘、失效重建、双路融合这些机制。用它跑出的指标不能写进任何结论。
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from app.schemas import Chunk

if TYPE_CHECKING:  # 只在类型检查时需要，运行时不引入 numpy（切块用不到）
    import numpy as np

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MANIFEST_VERSION = 1


# --------------------------------------------------------------------------- 切块


def split_sections(text: str) -> list[tuple[str, str]]:
    """按 `## ` 标题切节，返回 [(heading, body)]。开头无标题的部分归到 "(preamble)"。"""
    sections: list[tuple[str, list[str]]] = [("(preamble)", [])]
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(lines).strip()) for h, lines in sections]


def chunk_text(source: str, text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """切块。`chunk_id = "{source}#{序号}"`，序号在**该文件内**连续，不跨文件。"""
    if size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap，否则滑动窗口不前进")
    chunks: list[Chunk] = []
    for heading, body in split_sections(text):
        if not body:
            continue
        start = 0
        while start < len(body):
            piece = body[start : start + size].strip()
            if piece:
                chunks.append(
                    Chunk(chunk_id=f"{source}#{len(chunks):03d}", source=source, heading=heading, text=piece)
                )
            if start + size >= len(body):
                break
            start += size - overlap
    return chunks


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """文件名（去扩展名）即 `source`，与 `corpus.lock.json` 的 slug 对齐。"""
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"{corpus_dir} 下没有语料。先执行 python scripts/fetch_corpus.py。")
    return {f.stem: f.read_text(encoding="utf-8") for f in files}


def chunk_corpus(corpus_dir: Path, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in load_corpus(corpus_dir).items():
        chunks.extend(chunk_text(source, text, size, overlap))
    return chunks


# ------------------------------------------------------------------------ 向量化


class Embedder(Protocol):
    name: str
    dim: int | None

    def embed(self, texts: list[str]) -> "np.ndarray": ...


_WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def embed_text(chunk: Chunk) -> str:
    """送进 embedding 的文本。带上 heading：论文的章节名常含关键实体（如 "CRAG"）。"""
    return f"{chunk.heading}\n{chunk.text}"


class HashEmbedder:
    """确定性词袋哈希向量（sqrt-tf + L2 归一化，符号哈希降冲突偏置）。

    刻意做成**有意义的词面相似度**而不是随机噪声：随机向量会让「索引写对没有」永远
    测不出来（怎么跑都像在工作）。代价是没有同义与语序信息，所以只用于机制验证。
    """

    dim: int

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hash:{dim}"

    def embed(self, texts: list[str]) -> "np.ndarray":
        import numpy as np

        mat = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word, n in Counter(tokenize(text)).items():
                h = hashlib.blake2b(word.encode(), digest_size=4).digest()
                idx = int.from_bytes(h[:2], "big") % self.dim
                sign = 1.0 if h[2] & 1 else -1.0
                mat[row, idx] += sign * (1.0 + math.log(n))
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / (norms + 1e-9)


class OpenAIEmbedder:
    """真实语义向量。批量 32；`max_retries=2` 交给 SDK（索引是长任务，不因一次抖动全废）。"""

    dim: int | None = None

    def __init__(self, base_url: str, api_key: str, model: str, batch: int = 32) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=2)
        self.model = model
        self.batch = batch
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> "np.ndarray":
        import numpy as np

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + self.batch])
            out.extend(d.embedding for d in resp.data)
        mat = np.asarray(out, dtype="float32")
        self.dim = int(mat.shape[1])
        return mat


# ---------------------------------------------------------------------------- 落盘


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _fingerprint(corpus_dir: Path) -> dict[str, str]:
    """用 sha1 而不是 mtime：`git checkout` 会改 mtime 却不改内容，不该触发重建。"""
    return {p.stem: _sha1(p) for p in sorted(corpus_dir.glob("*.txt"))}


# ---------------------------------------------------------------------- 关键词侧

_TOKENIZER = "word-v2"


def tokenize(text: str) -> list[str]:
    """英文词级分词：小写 → 取 `[a-z0-9]` 词与带连字符/小数点的复合词（ADR-006 修订）。

    不做词形还原：`retrieval` 与 `retrieve` 是两个 token，召回会漏一点，但换来「无词典、
    无版本偏差」，且 `4,409`、`IsUse`、`HotpotQA` 这类字面量原样保留——这正是向量侧最糊、
    关键词侧最准的部分（源实现用 ES 的 ik_max_word 也是为这个）。

    千分位逗号先去掉再切：`4,409` 若切成 `4` 与 `409`，两段都是高频噪声，字面量优势就没了。
    """
    return _WORD.findall(re.sub(r"(?<=\d),(?=\d)", "", text.lower()))


def build_bm25(chunks: list[Chunk]):
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(embed_text(c)) for c in chunks])


def load_bm25(path: Path):
    """读回 BM25。`rank_bm25` 的对象可直接 pickle，省掉每次启动重建（1357 条要 1 秒级）。"""
    return pickle.loads(path.read_bytes())


def make_embedder(cfg) -> Embedder:
    kind = cfg.index.embedder
    if kind == "hash":
        return HashEmbedder(cfg.index.embed_dim)
    if kind == "openai":
        return OpenAIEmbedder(cfg.base_url, cfg.api_key, cfg.embed_model)
    raise ValueError(f"未知 embedder: {kind!r}，只支持 'hash' / 'openai'")


def build_index(cfg, embedder: Embedder | None = None) -> list[Chunk]:
    """构建并落盘。返回 chunks，调用方不必再读一次盘。"""
    import numpy as np

    corpus_dir = cfg.path(cfg.index.corpus_dir)
    index_dir = cfg.path(cfg.index.dir)
    size, overlap = cfg.retrieval.chunk_size, cfg.retrieval.chunk_overlap

    chunks = chunk_corpus(corpus_dir, size, overlap)
    emb = embedder or make_embedder(cfg)
    mat = emb.embed([embed_text(c) for c in chunks])
    if mat.shape[0] != len(chunks):
        raise RuntimeError(f"embedding 行数 {mat.shape[0]} 与片段数 {len(chunks)} 不一致")

    index_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(index_dir / "vectors.npz", vectors=mat)
    (index_dir / "bm25.pkl").write_bytes(pickle.dumps(build_bm25(chunks)))
    (index_dir / "chunks.jsonl").write_text("\n".join(c.model_dump_json() for c in chunks) + "\n", encoding="utf-8")
    manifest = {
        "version": MANIFEST_VERSION,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "embedder": emb.name,
        "tokenizer": _TOKENIZER,
        "dim": int(mat.shape[1]),
        "chunks": len(chunks),
        "files": _fingerprint(corpus_dir),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return chunks


def index_is_fresh(cfg, embedder: Embedder | None = None) -> bool:
    """索引是否存在，且与「当前语料 + 当前切块参数 + 当前 embedder」一致。"""
    index_dir = cfg.path(cfg.index.dir)
    need = ("manifest.json", "vectors.npz", "bm25.pkl", "chunks.jsonl")
    if not all((index_dir / n).exists() for n in need):
        return False
    try:
        m = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    want = {
        "version": MANIFEST_VERSION,
        "chunk_size": cfg.retrieval.chunk_size,
        "chunk_overlap": cfg.retrieval.chunk_overlap,
        "embedder": (embedder or make_embedder(cfg)).name,
        "tokenizer": _TOKENIZER,
    }
    if any(m.get(k) != v for k, v in want.items()):
        return False
    return m.get("files") == _fingerprint(cfg.path(cfg.index.corpus_dir))


def load_index(cfg) -> tuple[list[Chunk], "np.ndarray"]:
    """读索引；缺失或失效则重建（「一条命令跑通」的前提）。"""
    import numpy as np

    index_dir = cfg.path(cfg.index.dir)
    if not index_is_fresh(cfg):
        chunks = build_index(cfg)
    else:
        chunks = [
            Chunk.model_validate_json(line)
            for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    mat = np.load(index_dir / "vectors.npz")["vectors"]
    if mat.shape[0] != len(chunks):
        raise RuntimeError(f"索引损坏：向量 {mat.shape[0]} 行，片段 {len(chunks)} 条。删除 .index/ 重建。")
    return chunks, mat
