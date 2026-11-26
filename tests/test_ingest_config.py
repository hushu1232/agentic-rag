"""D09 / D12：切块幂等、配置拒绝拼错、索引度量一致。

`chunk_id` 是评测标注的唯一锚点。切块只要有一点不确定性（随机、依赖文件顺序、
带时间戳），`relevant_chunk_ids` 第二天就全部失效，而且失效是静默的——recall 会
莫名其妙下降到 0。所以这里把"同一语料切两次必须逐字节相同"写成断言。
"""

from __future__ import annotations

import pytest

from app.config import ConfigError, load_config
from app.index import chunk_corpus


@pytest.fixture(scope="module")
def corpus_dir(base_cfg):
    d = base_cfg.path(base_cfg.index.corpus_dir)
    if not any(d.glob("*.txt")):
        pytest.skip("语料未抓取（先跑 scripts/fetch_corpus.py）")
    return d


def test_chunking_is_idempotent(corpus_dir):
    first = chunk_corpus(corpus_dir)
    second = chunk_corpus(corpus_dir)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second], "同一语料两次切块内容必须一致"


def test_chunk_ids_are_unique(corpus_dir):
    chunks = chunk_corpus(corpus_dir)
    ids = [c.chunk_id for c in chunks]

    assert len(ids) == len(set(ids)), "chunk_id 重复会让评测标注指向两个片段"
    assert all(c.chunk_id.startswith(c.source + "#") for c in chunks), "id 必须能反解出 source"


def test_chunks_cover_whole_corpus(corpus_dir):
    """拼回去要覆盖原文（允许空白差异），否则有内容没进索引、永远检索不到。"""
    chunks = chunk_corpus(corpus_dir)
    by_source: dict[str, int] = {}
    for c in chunks:
        by_source[c.source] = by_source.get(c.source, 0) + len(c.text)

    for f in sorted(corpus_dir.glob("*.txt")):
        total = len(f.read_text(encoding="utf-8"))
        covered = by_source.get(f.stem, 0)
        assert covered >= total * 0.9, f"{f.stem} 只有 {covered}/{total} 字进了索引"


def test_config_rejects_unknown_top_level_key(tmp_path, base_cfg):
    """拼错的键等于那一整段配置静默失效——必须在启动时报错。"""
    text = base_cfg.path("config.yaml").read_text(encoding="utf-8").replace("retrieval:", "retrival:", 1)
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL_NAME"):
        p.with_name(".env").write_text("OPENAI_API_KEY=k\nOPENAI_BASE_URL=u\nMODEL_NAME=m\n", encoding="utf-8")

    with pytest.raises(ConfigError) as ei:
        load_config(p)
    assert "retrival" in str(ei.value)


def test_config_rejects_missing_gate(tmp_path, base_cfg):
    """少一个 gate 等于少一道闸，必须报错而不是当作"这项不设限"。"""
    text = base_cfg.path("config.yaml").read_text(encoding="utf-8").replace("    context_recall: 0.70\n", "")
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    p.with_name(".env").write_text("OPENAI_API_KEY=k\nOPENAI_BASE_URL=u\nMODEL_NAME=m\n", encoding="utf-8")

    with pytest.raises(ConfigError) as ei:
        load_config(p)
    assert "context_recall" in str(ei.value)


def test_embedder_choice_is_explicit(base_cfg):
    """embedder 必须显式声明：hash 与 openai 的指标不可横比，混用等于自欺。"""
    assert base_cfg.index.embedder in {"hash", "openai"}
    with pytest.raises(Exception):
        base_cfg.index.__class__(embedder="magic")
