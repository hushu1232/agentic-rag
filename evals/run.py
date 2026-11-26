"""评测入口。三件事：跑一遍数据集、算指标、过门禁。

四个决定值得说明：

1. `context_recall` 是**确定性**的（集合覆盖），不花一次模型调用。它只在"确实该检索"的
   样本上算：`expected_strategy=simple` 的题没有标注片段，硬算会除以零，也会奖励
   "多判 simple"——所以路由能力单独由 `routing_accuracy` 度量。
2. 判官失败记 `None` 并计入 `null_rate`，不当 0 分（见 `judges.py`）。
3. `--baseline` 默认取 `reports/` 里最新一份，`--gate` 才有意义：没有基线的门禁只是
   一个绝对值，看不出改一动是变好还是变坏。
4. `--ablation` 的四个变体**每次只加一个开关**（向量 → +BM25 → +RRF → +多跳），
   否则分不清提升来自哪一步。

    python -m evals.run --limit 5 --gate
    python -m evals.run --ablation --limit 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from app.config import Config, load_config
from app.graph import build_graph
from app.llm import Llm, LlmError
from app.nodes import REFUSAL
from app.rerank import Reranker
from app.retrieve import RetrievalError, Retriever
from app.schemas import State

from evals import report
from evals.judges import Judges

ABLATIONS = [
    ("A 单路向量", {"hybrid": False, "fusion": "rrf", "multihop": False}),
    ("B +BM25（拼接）", {"hybrid": True, "fusion": "concat", "multihop": False}),
    ("C +RRF", {"hybrid": True, "fusion": "rrf", "multihop": False}),
    ("D +多跳", {"hybrid": True, "fusion": "rrf", "multihop": True}),
]


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def variant(cfg: Config, **switches) -> Config:
    return cfg.model_copy(
        deep=True,
        update={
            "retrieval": cfg.retrieval.model_copy(update={k: v for k, v in switches.items() if k != "multihop"}),
            "graph": cfg.graph.model_copy(update={"multihop": switches.get("multihop", cfg.graph.multihop)}),
        },
    )


def evaluate(
    cfg: Config, items: list[dict], judges: Judges, review_refusal: bool = True, llm=None
) -> tuple[dict, list[dict]]:
    """`llm` 可注入假模型：回归测试与 `--limit` 冒烟都不该消耗 token。"""
    retriever = Retriever.from_config(cfg)
    llm = llm if llm is not None else Llm(cfg)
    graph = build_graph(cfg, llm, retriever, Reranker(cfg))
    rows: list[dict] = []

    for item in items:
        init = State(question=item["question"], max_rounds=cfg.graph.max_rounds)
        raw = graph.invoke(init, config={"recursion_limit": 4 * cfg.graph.max_rounds + 6})
        st = raw if isinstance(raw, State) else State.model_validate(raw)

        selected = {c.chunk_id for c in st.selected}
        relevant = set(item["relevant_chunk_ids"])
        recall = len(selected & relevant) / len(relevant) if relevant else None

        is_refusal = st.answer.strip().startswith(REFUSAL)
        if item["type"] == "unanswerable":
            refusal_ok = is_refusal
            if refusal_ok and review_refusal:
                verdict = judges.refusal_reviewed(item["question"], st.answer)
                refusal_ok = True if verdict is None else verdict  # 判官失败不改规则结论
        else:
            refusal_ok = not is_refusal

        grounded = judges.groundedness(item["question"], st.answer, st.context) if st.selected else None
        relevant_score = judges.relevance(item["question"], st.answer)

        flags: list[str] = []
        if st.strategy != item["expected_strategy"]:
            flags.append(f"routing(expected={item['expected_strategy']}, got={st.strategy})")
        if not refusal_ok:
            flags.append("refusal")
        if recall is not None and recall < 1.0:
            flags.append(f"recall={recall:.2f}")
        if grounded is not None and grounded < 1.0:
            flags.append(f"grounded={grounded:.2f}")

        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "answer": st.answer,
                "strategy": st.strategy.value if st.strategy else None,
                "expected_strategy": item["expected_strategy"],
                "selected": sorted(selected),
                "relevant": sorted(relevant),
                "recall": recall,
                "groundedness": grounded,
                "answer_relevance": relevant_score,
                "refusal_ok": refusal_ok,
                "degraded": st.degraded,
                "tokens": st.tokens or llm.tokens,
                "trace": [t.model_dump() for t in st.trace],
                "flags": flags,
            }
        )

    metrics = {
        "context_recall": mean([r["recall"] for r in rows]),
        "groundedness": mean([r["groundedness"] for r in rows]),
        "answer_relevance": mean([r["answer_relevance"] for r in rows]),
        "refusal_accuracy": mean([1.0 if r["refusal_ok"] else 0.0 for r in rows]),
        "routing_accuracy": mean([1.0 if r["strategy"] == r["expected_strategy"] else 0.0 for r in rows]),
        "null_rate": (judges.null / judges.total) if judges.total else 0.0,
        "n": len(rows),
        "n_context": sum(1 for r in rows if r["recall"] is not None),
        "tokens": sum(r["tokens"] for r in rows),
    }
    return metrics, rows


def gate_failures(metrics: dict, gates: dict[str, float]) -> list[tuple[str, float | None, float]]:
    fails = []
    for key, thr in gates.items():
        value = metrics.get(key)
        if value is None:
            fails.append((key, None, thr))
        elif (value > thr) if key == "null_rate" else (value < thr):
            fails.append((key, value, thr))
    return fails


def gate_lines(metrics: dict, gates: dict[str, float]) -> list[str]:
    """门禁失败的可读说明。

    抽成纯函数就为了能断言「还差几个样本」算得对——一个裸差值 0.0200 看不出是
    「差一条」还是「差得远」，换算成样本数才指导得了下一步该修什么。
    """
    lines: list[str] = []
    for key, value, thr in gate_failures(metrics, gates):
        if value is None:
            lines.append(f"  ✗ {key}: 没有有效值（判官全部失败，或分母为 0）")
            continue
        gap = abs(thr - value)
        # context_recall 的分母是有标注的样本数；null_rate 的分母是判官调用数，不换算成样本。
        denom = metrics.get("n_context") if key == "context_recall" else metrics.get("n", 0)
        need = ""
        if key != "null_rate" and denom:
            need = f"，还差约 {math.ceil(gap * denom - 1e-9)} 个样本"
        lines.append(f"  ✗ {key}: {value:.4f}（阈值 {thr:.4f}，差 {gap:.4f}{need}）")
    return lines


def latest_report(reports_dir: Path, exclude: Path | None = None) -> Path | None:
    """默认基线 = 最新一份**普通评测**报告。

    必须排除消融报告：它存的是四组 `variants` 而不是单组 `metrics`，拿它当基线会算出
    一列 `—`（键不存在），看起来像"指标全丢了"而不是"选错了基线"。
    """
    files = sorted(
        p for p in reports_dir.glob("*.json") if p != exclude and not p.stem.endswith("-ablation")
    )
    return files[-1] if files else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.run", description="跑评测集、算指标、过门禁")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    ap.add_argument("--gate", action="store_true", help="按 config 的阈值判失败并退出 1")
    ap.add_argument("--baseline", metavar="PATH", help="对比基线 json（默认取 reports/ 最新）")
    ap.add_argument("--ablation", action="store_true", help="跑四组消融")
    ap.add_argument("--label", default="eval")
    ap.add_argument("--no-review", action="store_true", help="不调判官复核拒答（省调用）")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    # 基线路径在**跑之前**校验：跑完 30 条、烧掉几百次判官调用才发现路径打错，
    # 那是纯浪费。这里提前失败，且退出码独立于门禁失败（2 vs 1）。
    if args.baseline and not Path(args.baseline).exists():
        print(f"[baseline] 找不到基线文件：{args.baseline}\n"
              f"           先跑一次不带 --baseline 的评测生成基线，或检查路径。", file=sys.stderr)
        return 2

    ds_path = cfg.path(cfg.evals.dataset)
    items = load_dataset(ds_path)
    if args.limit:
        items = items[: args.limit]
    dataset_sha1 = hashlib.sha1(ds_path.read_bytes()).hexdigest()

    def meta(c: Config, label: str) -> dict:
        return {
            "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "embedder": c.index.embedder,
            "model": c.model,
            "judge_model": c.judge_model,
            "dataset": str(ds_path),
            "dataset_sha1": dataset_sha1,
            "reports_dir": str(c.path(c.evals.reports_dir)),
            "label": label,
        }

    if args.ablation:
        if args.gate:
            # 消融的目的是看趋势，不是过闸：每组用的是同一套阈值，中途 FAIL 只会打断对比。
            print("[ablation] 已忽略 --gate（消融看趋势，门禁请单独跑一次不含 --ablation 的评测）",
                  file=sys.stderr)
        results = []
        try:
            for label, switches in ABLATIONS:
                c = variant(cfg, **switches)
                judges = Judges(c)
                metrics, _ = evaluate(c, items, judges, review_refusal=not args.no_review)
                results.append({"label": label, "metrics": metrics})
                print(f"  {label:<16} recall={report._fmt(metrics['context_recall'])} "
                      f"grounded={report._fmt(metrics['groundedness'])}", file=sys.stderr)
        except LlmError as exc:
            print(f"[llm] 消融中断（已完成 {len(results)}/4 组）：{exc}\n"
                  f"      检查 .env 的 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME。", file=sys.stderr)
            # 已经跑出来的组仍然落盘——半张表有诊断价值，扔掉等于白花钱。
            if results:
                json_path, md_path = report.save_ablation(results, meta(cfg, "ablation-partial"), cfg.evals.gates)
                print(f"\n已完成 {len(results)} 组的报告：{md_path}", file=sys.stderr)
            return 3
        json_path, md_path = report.save_ablation(results, meta(cfg, "ablation"), cfg.evals.gates)
        print("\n| 变体 | context_recall | Δ | groundedness | Δ |")
        print("|---|---:|---:|---:|---:|")
        for r in report.ablation_rows(results):
            print("| " + " | ".join(r) + " |")
        print(f"\n报告：{json_path}\n      {md_path}")
        return 0

    judges = Judges(cfg)
    try:
        metrics, rows = evaluate(cfg, items, judges, review_refusal=not args.no_review)
    except LlmError as exc:
        # 没配 key 时最常见的失败点。给一句人话 + 专用退出码，不要甩 traceback。
        print(f"[llm] 评测中断：{exc}\n"
              f"      检查 .env 的 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME，"
              f"或先用 --limit 1 确认链路。", file=sys.stderr)
        return 3
    except RetrievalError as exc:
        print(f"[retrieval] 评测中断：{exc}\n"
              f"      索引可能未建或已失效，先跑 python -m app.cli --reindex。", file=sys.stderr)
        return 4
    json_path, md_path = report.save(metrics, rows, meta(cfg, args.label), cfg.evals.gates, args.label)

    print(f"样本 {metrics['n']} 条｜参与 recall {metrics['n_context']} 条｜tokens {metrics['tokens']}")
    for key, name, _ in report.METRIC_NAMES:
        print(f"  {name:<18} {report._fmt(metrics.get(key))}")

    cur_meta = meta(cfg, args.label)
    base_path = Path(args.baseline) if args.baseline else latest_report(Path(cfg.path(cfg.evals.reports_dir)), json_path)

    if base_path and base_path.exists():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        if "metrics" not in base:
            print(f"  ⚠ {base_path.name} 不是单组评测报告（没有 metrics 键，可能是消融报告），跳过 diff",
                  file=sys.stderr)
        else:
            base_emb = base.get("meta", {}).get("embedder")
            if base_emb and base_emb != cur_meta["embedder"]:
                print(f"  ⚠ 基线与本次 embedder 不同（{base_emb} → {cur_meta['embedder']}），Δ 不可比",
                      file=sys.stderr)
            print(f"\n对比基线 {base_path.name}")
            print("| 指标 | 基线 | 当前 | Δ |")
            print("|---|---:|---:|---:|")
            for name, b, c, d in report.diff_rows(base, {"metrics": metrics}):
                print(f"| {name} | {b} | {c} | {d} |")

    print(f"\n报告：{json_path}\n      {md_path}")

    if args.gate:
        fails = gate_failures(metrics, cfg.evals.gates)
        if fails:
            print("\n门禁未通过：", file=sys.stderr)
            for line in gate_lines(metrics, cfg.evals.gates):
                print(line, file=sys.stderr)
            return 1
        print("\n门禁全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
