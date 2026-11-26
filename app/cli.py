"""命令行入口。三类错误映射为三种退出码，stderr 给出可读原因。

`--ask` 是唯一必须「一条命令跑通」的路径：索引不存在时自动构建（首次会拉 embedding，
用 hash embedder 时不联网）。

退出码：
  0 正常 / 1 内容层面失败（留给评测门禁） / 2 配置无效 / 3 模型失败 / 4 检索失败
"""

from __future__ import annotations

import argparse
import sys

from app.config import ConfigError, load_config
from app.graph import run
from app.index import build_index, index_is_fresh
from app.llm import Llm, LlmError
from app.rerank import Reranker
from app.retrieve import RetrievalError, Retriever

EXIT = {"config": 2, "llm": 3, "retrieval": 4}


def _fail(kind: str, exc: Exception) -> int:
    print(f"[{kind}] {exc}", file=sys.stderr)
    return EXIT[kind]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="app.cli", description="Agentic RAG over 14 RAG papers")
    ap.add_argument("--ask", metavar="Q", help="提问")
    ap.add_argument("--verbose", action="store_true", help="打印每个节点的耗时与决策")
    ap.add_argument("--reindex", action="store_true", help="强制重建索引")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    if not args.ask and not args.reindex:
        ap.print_help()
        return 1

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        return _fail("config", exc)

    try:
        if args.reindex or not index_is_fresh(cfg):
            chunks = build_index(cfg)
            if args.verbose or args.reindex:
                print(f"索引已构建：{len(chunks)} 片段 → {cfg.path(cfg.index.dir)}", file=sys.stderr)
            if not args.ask:
                return 0
        retriever = Retriever.from_config(cfg)
    except FileNotFoundError as exc:
        return _fail("retrieval", exc)
    except Exception as exc:  # embedding 服务不可达等，归到检索层
        return _fail("retrieval", exc)

    try:
        state = run(cfg, Llm(cfg), retriever, Reranker(cfg), args.ask)
    except LlmError as exc:
        return _fail("llm", exc)
    except RetrievalError as exc:
        return _fail("retrieval", exc)

    print(state.answer)
    if args.verbose:
        print("\n--- trace ---", file=sys.stderr)
        for step in state.trace:
            print(f"{step.node:<10} {step.ms:>5}ms  {step.decision}  {step.detail}", file=sys.stderr)
        print(f"tokens={state.tokens} degraded={state.degraded}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
