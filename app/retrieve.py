"""混合召回：向量 + BM25 双路并行，RRF 按名次融合。

为什么不按分数加权融合：两路的分数根本不可比（余弦在 0–1，BM25 无上界、随语料规模漂移），
任何加权系数都是拍脑袋。RRF 只用名次，`1/(k+rank)` 与量纲无关——这也是源实现停在
"concat + 去重" 时缺的那一步（`hybrid-retrieval.mjs:71-89`）。

三处刻意设计：
- **两路独立发起**：一路抛错不影响另一路，降级信息进 `degraded`，由节点写进 trace。
- **分数不覆盖**：同一个 `chunk_id` 被两路各召回时，两个原始分都留下，只由 `rrf` 决定排序，
  否则事后无法诊断"是这个片段本来就近，还是融合把它抬上来的"。
- **入口接受一组 query**：源实现 hybrid 就是拿 4 条改写 query 并行召回；单条是它的特例，
  多查只是多几组排名，RRF 天然能合并（对应 `retrieval.query_augment` 开关）。
"""

from __future__ import annotations

import numpy as np

from app.index import load_bm25, load_index, make_embedder, tokenize
from app.schemas import Candidate, Chunk


class RetrievalError(RuntimeError):
    """两路都失败时才抛。单路失败属于降级，不抛。"""


def _as_candidate(chunk: Chunk) -> Candidate:
    return Candidate(**chunk.model_dump())


def _top_k(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in idx[np.argsort(-scores[idx])]]


def vector_search(query: str, chunks: list[Chunk], mat: np.ndarray, embedder, k: int) -> list[Candidate]:
    """`mat` 预先做过 L2 归一化，故内积即余弦。"""
    q = embedder.embed([query])[0]
    q = q / (np.linalg.norm(q) + 1e-9)
    scores = mat @ q
    return [(_as_candidate(chunks[i])).model_copy(update={"vector_score": float(scores[i])}) for i in _top_k(scores, k)]


def bm25_search(query: str, chunks: list[Chunk], bm25, k: int) -> list[Candidate]:
    scores = np.asarray(bm25.get_scores(tokenize(query)), dtype="float32")
    return [(_as_candidate(chunks[i])).model_copy(update={"bm25_score": float(scores[i])}) for i in _top_k(scores, k)]


def rrf_fuse(lists: list[list[Candidate]], k: int) -> list[Candidate]:
    """名次融合。同一片段出现多次只累加 `rrf`，两个原始分各自保留（先到者优先）。"""
    acc: dict[str, Candidate] = {}
    for ranked in lists:
        for rank, cand in enumerate(ranked, start=1):
            cur = acc.get(cand.chunk_id)
            if cur is None:
                cur = cand.model_copy(update={"rrf": 0.0})
                acc[cand.chunk_id] = cur
            else:
                if cur.vector_score is None:
                    cur.vector_score = cand.vector_score
                if cur.bm25_score is None:
                    cur.bm25_score = cand.bm25_score
            cur.rrf += 1.0 / (k + rank)
    # 平票必须有确定的次序，否则同一输入两次运行的结果可能不同（评测就无法 diff）。
    # 平票用 chunk_id 兜底，它只依赖语料，不依赖运行环境。
    return sorted(acc.values(), key=lambda c: (-c.rrf, c.chunk_id))


def concat_fuse(lists: list[list[Candidate]]) -> list[Candidate]:
    """两路直接拼接、按 `chunk_id` 去重保留先到者——**不做任何融合**。

    这不是"简化版 RRF"，而是源实现的实际形态（`hybrid-retrieval.mjs:71-89`：`[...esDocs, ...milvusDocs]`
    后去重，向量路的相似度完全没进排序）。保留它作为消融 B 组：只有拿它当基线，
    "RRF 到底值多少个百分点"才是可测的，而不是靠感觉。
    """
    seen: set[str] = set()
    out: list[Candidate] = []
    for ranked in lists:
        for cand in ranked:
            if cand.chunk_id in seen:
                continue
            seen.add(cand.chunk_id)
            out.append(cand)
    return out


class Retriever:
    def __init__(self, chunks: list[Chunk], mat: np.ndarray, bm25, embedder, cfg) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.cfg = cfg
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.mat = mat / (norms + 1e-9)
        self.bm25 = bm25

    @classmethod
    def from_config(cls, cfg) -> "Retriever":
        chunks, mat = load_index(cfg)
        bm25 = load_bm25(cfg.path(cfg.index.dir) / "bm25.pkl")
        return cls(chunks, mat, bm25, make_embedder(cfg), cfg)

    def search(
        self,
        queries: list[str],
        k: int | None = None,
        hybrid: bool | None = None,
        fusion: str | None = None,
    ) -> tuple[list[Candidate], list[str]]:
        """返回 (融合后的候选, 降级标记)。两路全失败才抛 `RetrievalError`。"""
        k = k or self.cfg.retrieval.top_k_each
        hybrid = self.cfg.retrieval.hybrid if hybrid is None else hybrid
        fusion = self.cfg.retrieval.fusion if fusion is None else fusion
        if not queries:
            raise RetrievalError("没有可检索的查询")

        ranked: list[list[Candidate]] = []
        degraded: list[str] = []
        last_error: Exception | None = None

        for q in queries:
            try:
                ranked.append(vector_search(q, self.chunks, self.mat, self.embedder, k))
            except Exception as exc:  # 单路失败 → 记降级，继续
                degraded.append("retrieval_degraded:vector")
                last_error = exc
            if hybrid:
                try:
                    ranked.append(bm25_search(q, self.chunks, self.bm25, k))
                except Exception as exc:
                    degraded.append("retrieval_degraded:bm25")
                    last_error = exc

        if not ranked:
            raise RetrievalError(f"两路召回均失败：{last_error}") from last_error
        fused = rrf_fuse(ranked, self.cfg.retrieval.rrf_k) if fusion == "rrf" else concat_fuse(ranked)
        return fused, sorted(set(degraded))
