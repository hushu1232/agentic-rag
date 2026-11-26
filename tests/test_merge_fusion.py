"""D07 / D08：融合必须真的融合，去重键不许兜底折叠。

源实现的 `merge` 是 `[...esDocs, ...milvusDocs]` 后按 id 去重，`docFromEsHit` 不写
`score`——两路召回**没有融合**，向量相似度完全没参与排序。另外去重键写成
`doc.id ?? supplier ?? 'unknown'`，会让一批没有 id 的文档塌成一条。

这里同时钉住三件事：同 id 只出现一次、两个原始分都保留、双路命中的排名高于单路。
"""

from __future__ import annotations

from app.retrieve import concat_fuse, rrf_fuse
from app.schemas import Candidate


def cand(cid: str, vec: float | None = None, bm25: float | None = None) -> Candidate:
    return Candidate(chunk_id=cid, source=cid.split("#")[0], heading="h", text="t",
                     vector_score=vec, bm25_score=bm25)


def test_rrf_ranks_double_hits_above_single():
    vector = [cand("a#1", vec=0.9), cand("a#2", vec=0.8)]
    bm25 = [cand("a#2", bm25=12.0), cand("a#3", bm25=9.0)]
    fused = rrf_fuse([vector, bm25], k=60)

    order = [c.chunk_id for c in fused]
    assert order[0] == "a#2", "两路都命中的片段必须排第一"
    assert fused[0].rrf > fused[1].rrf
    assert len({c.chunk_id for c in fused}) == 3, "同 id 只能出现一次"


def test_rrf_keeps_both_raw_scores():
    fused = rrf_fuse([[cand("a#1", vec=0.9)], [cand("a#1", bm25=7.5)]], k=60)

    assert fused[0].vector_score == 0.9, "融合不许覆盖向量分——事后要能诊断是谁把它抬上来的"
    assert fused[0].bm25_score == 7.5


def test_rrf_is_deterministic_on_ties():
    """两路互为镜像时 RRF 分完全相同，必须有确定的次序，否则两次运行结果不同、diff 失效。"""
    left = [cand("a#1", vec=0.9), cand("a#2", vec=0.8)]
    right = [cand("b#1", bm25=9.0), cand("b#2", bm25=8.0)]
    mirror_l = [cand("x#1", vec=0.9), cand("x#2", vec=0.8)]
    mirror_r = [cand("x#2", bm25=9.0), cand("x#1", bm25=8.0)]

    a = [c.chunk_id for c in rrf_fuse([left, right], k=60)]
    b = [c.chunk_id for c in rrf_fuse([mirror_l, mirror_r], k=60)]
    assert a == [c.chunk_id for c in rrf_fuse([left, right], k=60)], "同一输入必须得到同一顺序"
    assert len(b) == 2


def test_concat_does_not_fuse():
    """消融 B 组：拼接去重保留先到者，向量路的分数不参与排序——这是源实现的形态。"""
    vector = [cand("a#9", vec=0.99), cand("a#1", vec=0.10)]
    bm25 = [cand("a#1", bm25=20.0), cand("a#8", bm25=18.0)]
    out = concat_fuse([vector, bm25])

    assert [c.chunk_id for c in out] == ["a#9", "a#1", "a#8"], "顺序必须是'先到先得'"
    assert all(c.rrf == 0.0 for c in out), "拼接不做融合，rrf 必须保持 0"


def test_duplicate_ids_without_source_do_not_collapse():
    """D07：去重键只能是 chunk_id，不能兜底成常量，否则不同文档会塌成一条。"""
    a = Candidate(chunk_id="doc-a#0", source="doc-a", heading="h", text="t")
    b = Candidate(chunk_id="doc-b#0", source="doc-b", heading="h", text="t")
    out = concat_fuse([[a, b]])

    assert len(out) == 2, "两条不同来源的片段必须都保留"


def test_empty_lists_return_empty():
    assert rrf_fuse([], k=60) == []
    assert concat_fuse([]) == []
    assert concat_fuse([[], []]) == []


def test_search_actually_uses_the_chosen_fusion(cfg, monkeypatch):
    """只测 `rrf_fuse` 是不够的：把 `search` 里的融合换成拼接，纯函数测试照样绿。

    这条走 `Retriever.search`，断言两种模式给出**不同**的排序——这才证明开关接进了主路径。
    """
    import numpy as np

    import app.retrieve as rv
    from app.retrieve import Retriever

    monkeypatch.setattr(rv, "vector_search", lambda q, *a, **k: [cand("a#1", vec=0.9), cand("a#2", vec=0.8)])
    monkeypatch.setattr(rv, "bm25_search", lambda q, *a, **k: [cand("a#2", bm25=9.0), cand("a#3", bm25=7.0)])

    r = Retriever([], np.zeros((0, 8)), None, None, cfg)
    rrf_out, _ = r.search(["q"], fusion="rrf")
    concat_out, _ = r.search(["q"], fusion="concat")

    assert rrf_out[0].chunk_id == "a#2", "RRF 下双路命中必须排第一"
    assert [c.chunk_id for c in concat_out] == ["a#1", "a#2", "a#3"], "拼接是先到先得，不看融合分"
    assert [c.chunk_id for c in rrf_out] != [c.chunk_id for c in concat_out], "两种模式必须真的不同"
