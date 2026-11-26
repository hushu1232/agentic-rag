"""配置层：config.yaml + 环境变量 → 带校验的 Config。

密钥与模型名只来自环境变量（`.env` 自行解析，不引 python-dotenv）；数值参数只来自
config.yaml。任何拼写错误在启动时炸掉，不靠默认值掩盖——源实现里字段名打错却静默
跑通（`routeReson`、`retrieveDocs`）正是靠这一层拦住的。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SECTIONS = ("retrieval", "rerank", "graph", "index", "evals")
GATE_KEYS = ("context_recall", "groundedness", "answer_relevance",
             "refusal_accuracy", "routing_accuracy", "null_rate")


class ConfigError(Exception):
    """配置缺失或非法。CLI 映射为退出码 1，提示语直接给出变量名或文件位置。"""


def _load_dotenv(path: Path) -> None:
    """最小 .env 解析：KEY=VALUE、# 注释、不覆盖已存在的环境变量。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalCfg(_Strict):
    chunk_size: int = Field(800, ge=100, le=2000)
    chunk_overlap: int = Field(100, ge=0, le=500)
    top_k_each: int = Field(8, ge=1, le=50)
    top_n: int = Field(5, ge=1, le=20)
    rrf_k: int = Field(60, ge=1, le=200)
    hybrid: bool = True
    # 消融用：rrf = 名次融合；concat = 两路直接拼接去重（源实现的形态，对照基线）
    fusion: Literal["rrf", "concat"] = "rrf"
    query_augment: bool = False


class RerankCfg(_Strict):
    enabled: bool = False
    model: str = "qwen3-rerank"
    url: str = ""


class GraphCfg(_Strict):
    multihop: bool = True
    max_rounds: int = Field(3, ge=1, le=10)
    max_sub_questions: int = Field(6, ge=1, le=10)


class IndexCfg(_Strict):
    corpus_dir: str = "data/corpus"
    dir: str = ".index"
    # ADR-015：hash 是无 key 时的确定性占位，保证落盘/失效/融合逻辑可被测试；
    # openai 才产生语义向量，指标只认它。
    embedder: Literal["hash", "openai"] = "hash"
    embed_dim: int = Field(256, ge=32, le=4096)


class EvalsCfg(_Strict):
    dataset: str = "evals/dataset.jsonl"
    reports_dir: str = "reports"
    gates: dict[str, float]


class Config(_Strict):
    root: Path
    base_url: str
    api_key: str
    model: str
    embed_model: str
    judge_model: str
    rerank_api_key: str
    retrieval: RetrievalCfg
    rerank: RerankCfg
    graph: GraphCfg
    index: IndexCfg
    evals: EvalsCfg

    def path(self, rel: str | Path) -> Path:
        """配置里的路径一律相对项目根，避免「在哪个目录执行」影响结果。"""
        p = Path(rel)
        return p if p.is_absolute() else self.root / p


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"缺少环境变量 {name}。复制 .env.example 为 .env 后填入，或 export {name}=..."
        )
    return value


def load_config(path: str | Path = "config.yaml") -> Config:
    root = Path(path).resolve().parent
    _load_dotenv(root / ".env")
    if not (root / "config.yaml").exists():
        raise ConfigError(f"找不到配置文件 {root / 'config.yaml'}")
    raw: Any = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml 顶层必须是映射（key: value）")

    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ConfigError(f"config.yaml 存在未知顶层字段：{sorted(unknown)}，请检查拼写")

    try:
        evals_raw = dict(raw.get("evals", {}))
        gates = evals_raw.pop("gates", {})
        sections = {
            "retrieval": RetrievalCfg(**raw.get("retrieval", {})),
            "rerank": RerankCfg(**raw.get("rerank", {})),
            "graph": GraphCfg(**raw.get("graph", {})),
            "index": IndexCfg(**raw.get("index", {})),
            "evals": EvalsCfg(gates=gates, **evals_raw),
        }
    except ValidationError as exc:
        raise ConfigError(f"config.yaml 字段非法：\n{exc}") from exc

    missing = [k for k in GATE_KEYS if k not in sections["evals"].gates]
    if missing:
        raise ConfigError(f"evals.gates 缺少指标 {missing}（拼错的键等于该项不设闸）")
    bad = {k: v for k, v in sections["evals"].gates.items() if not 0.0 <= v <= 1.0}
    if bad:
        raise ConfigError(f"evals.gates 阈值必须在 0–1 之间：{bad}")

    rerank: RerankCfg = sections["rerank"]  # type: ignore[assignment]
    rerank_key = os.environ.get("RERANK_API_KEY", "").strip()
    if rerank.enabled and not rerank_key:
        raise ConfigError("rerank.enabled=true 但环境变量 RERANK_API_KEY 为空")
    if rerank.enabled and not rerank.url:
        raise ConfigError("rerank.enabled=true 但 config.yaml 的 rerank.url 为空")

    model = _env("MODEL_NAME")
    return Config(
        root=root,
        base_url=_env("OPENAI_BASE_URL"),
        api_key=_env("OPENAI_API_KEY"),
        model=model,
        embed_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-v3").strip(),
        judge_model=os.environ.get("JUDGE_MODEL", "").strip() or model,
        rerank_api_key=rerank_key,
        **sections,  # type: ignore[arg-type]
    )
