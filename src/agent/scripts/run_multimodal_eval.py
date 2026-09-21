#!/usr/bin/env python3
"""Multimodal gold-set evaluation runner (T2.5 step 2) — the four-step flow.

1. RETRIEVE: each question embeds → dense top-k over ``rag_multimodal``
   (``generate_answer=false`` semantics; no filters unless the question type
   dictates scope — we measure RAW retrieval plus a scope-violation rate).
2. HARD ASSERTIONS (pure computation, no LLM — the M4 lesson):
   - HitRate@k: any gold chunk in top-k (single_hop headline metric);
   - GoldRecall@k: |gold ∩ top-k| / |gold| (aggregation coverage);
   - MRR: 1/rank of the first gold hit (aggregation ordering quality);
   - scope_violation: hits landing outside the question's book scope.
3. SOFT SCORES (optional ``--ragas-multimodal``): RAGAS multimodal
   faithfulness/relevance on the retrieved contexts with the gold reference as
   the judged response (ragas@fc0d071 MultiModalFaithfulness/MultiModalRelevance,
   ark/deepseek judge via tools.llm.get_llm). Never gates: import failures or
   per-row scoring errors degrade to recorded skips.
4. GATE REPORT: per-metric means by question type, written next to the m4_bench
   pattern; ``--baseline`` compares against a previous report at ±threshold
   (default 2%) and flags regressions — HitRate/GoldRecall/MRR (higher=better)
   and scope_violation (error rate, lower=better: a baseline of 0.0 with an
   out-of-scope rate above the tolerance FAILs, review P2-5). RAGAS soft
   scores are report-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))


def load_evalset(path: Path, keep_leaks: bool) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") or []
    # Drop leak suspects unless explicitly kept (human curation hook).
    kept = [q for q in questions if keep_leaks or not q.get("leak_suspect")]
    return kept


def rank_metrics(hit_chunk_ids: list[str], gold: list[str], k: int) -> dict:
    top_k = hit_chunk_ids[:k]
    gold_set = set(gold)
    first_rank = next((i + 1 for i, cid in enumerate(top_k) if cid in gold_set), None)
    recall = len(gold_set & set(top_k)) / len(gold_set) if gold_set else 0.0
    return {
        "hit": 1.0 if first_rank else 0.0,
        "recall": recall,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


async def retrieve_all(
    questions: list[dict],
    top_k: int,
    collection: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    from tools.multimodal_vectorizer import build_vectorizer
    from tools.retrieval_backends.dense_milvus_multimodal import (
        MilvusMultimodalDenseBackend,
    )

    # collection=None keeps the CLI default (config.multimodal_collection); the
    # evaluate API passes the caller's dynamic collection through.
    say = progress or print
    backend = MilvusMultimodalDenseBackend(collection=collection)
    vectorizer = build_vectorizer()
    rows: list[dict] = []
    try:
        for i, question in enumerate(questions, 1):
            t0 = time.perf_counter()
            embed = await vectorizer.embed_text(question["question"])
            if embed.vector is None:
                rows.append({"question": question, "hits": [], "error": embed.error})
                continue
            # Sync pymilvus search → worker thread so the job never blocks the
            # API event loop when run_evaluation_core runs as a background job
            # (review P1-1).
            hits = await asyncio.to_thread(
                backend.search, embed.vector, limit=top_k, log_stage="mm_eval_retrieve"
            )
            rows.append(
                {
                    "question": question,
                    "hits": [
                        {
                            "chunk_id": h["chunk_id"],
                            "document_id": h["document_id"],
                            "book_id": h["book_id"],
                            "kind": h["kind"],
                            "score": h["score"],
                            "page_no": h["page_no"],
                            # Kept for the optional RAGAS multimodal soft scores
                            # (retrieved contexts); harmless otherwise.
                            "text_preview": h.get("text_preview"),
                        }
                        for h in hits
                    ],
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "error": None,
                }
            )
            say(f"  [{i}/{len(questions)}] {len(hits)} hits in {rows[-1]['latency_ms']}ms")
    finally:
        await vectorizer.aclose()
    return rows


def hard_assertions(rows: list[dict], top_k: int) -> list[dict]:
    results = []
    for row in rows:
        q = row["question"]
        hit_ids = [h["chunk_id"] for h in row["hits"]]
        metrics = rank_metrics(hit_ids, q["gold_chunk_ids"], top_k)
        # Scope assertion: hits outside the question's book set (cross_book = union scope).
        scope_books = set(q.get("scope_books") or ([q["scope"]["book_id"]] if q.get("scope", {}).get("book_id") else []))
        out_of_scope = 0
        if scope_books:
            out_of_scope = sum(1 for h in row["hits"] if (h["book_id"] or "") not in scope_books)
        results.append(
            {
                "type": q["type"],
                "question": q["question"],
                "metrics": metrics,
                "scope_violation": out_of_scope / len(row["hits"]) if row["hits"] and scope_books else 0.0,
                "top_k_hits": hit_ids[:top_k],
                "gold_chunk_ids": q["gold_chunk_ids"],
                "latency_ms": row.get("latency_ms"),
                "error": row.get("error"),
            }
        )
    return results


def _p95(latencies: list[float]) -> float | None:
    """95th percentile over the given (already filtered) latency list.

    The index must be based on the filtered list length — indexing with the
    unfiltered item count raised IndexError when embed-failure rows (no
    latency_ms) were present. Empty input (all rows failed) → None.
    """
    if not latencies:
        return None
    ordered = sorted(latencies)
    return round(ordered[int(0.95 * (len(ordered) - 1))], 1)


def _metric_panel(items: list[dict]) -> dict:
    """Summary panel for one question group.

    Embed-failure rows are excluded from the metric means (they previously
    dragged every score toward 0 as silent zeros) and counted separately in
    ``errors``; p95 latency is computed over successful rows only.
    """
    ok = [x for x in items if not x.get("error")]
    latencies = [x["latency_ms"] for x in ok if x.get("latency_ms") is not None]
    panel: dict = {
        "n": len(items),
        "errors": len(items) - len(ok),
    }
    if ok:
        panel["hit_rate"] = round(statistics.mean(x["metrics"]["hit"] for x in ok), 4)
        panel["gold_recall"] = round(statistics.mean(x["metrics"]["recall"] for x in ok), 4)
        panel["mrr"] = round(statistics.mean(x["metrics"]["mrr"] for x in ok), 4)
        panel["scope_violation"] = round(statistics.mean(x["scope_violation"] for x in ok), 4)
    else:
        panel["hit_rate"] = None
        panel["gold_recall"] = None
        panel["mrr"] = None
        panel["scope_violation"] = None
    panel["p95_latency_ms"] = _p95(latencies)
    return panel


def summarize(results: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    summary: dict[str, dict] = {}
    for qtype, items in sorted(by_type.items()):
        summary[qtype] = _metric_panel(items)
    if results:
        summary["overall"] = _metric_panel(results)
    return summary


def compare_baseline(summary: dict, baseline_path: Path, threshold: float) -> dict:
    """Regression gate vs a previous report at ±threshold (default 2%).

    hit_rate / gold_recall / mrr are higher-is-better: a delta below
    -threshold is a regression. scope_violation is an ERROR RATE
    (lower-is-better, review P2-5): a delta ABOVE +threshold is a regression —
    a baseline of 0.0 with any real out-of-scope leak beyond the tolerance
    fails the gate. RAGAS soft scores stay report-only.
    """
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_summary = baseline.get("summary") or {}
    regressions: list[dict] = []

    def _panel(metric: str) -> tuple[Any, Any]:
        return (summary.get("overall") or {}).get(metric), (base_summary.get("overall") or {}).get(metric)

    for metric in ("hit_rate", "gold_recall", "mrr"):
        current, previous = _panel(metric)
        if current is None or previous is None:
            continue
        delta = round(current - previous, 4)
        if delta < -threshold:
            regressions.append({"metric": metric, "previous": previous, "current": current, "delta": delta})
    current, previous = _panel("scope_violation")
    if current is None or previous is None:
        # Legacy baselines predate scope_violation — treat a missing baseline as
        # 0.0 so an out-of-scope rate in the current run still gates.
        previous = 0.0
    if current is not None:
        delta = round(current - previous, 4)
        if delta > threshold:
            regressions.append(
                {"metric": "scope_violation", "previous": previous, "current": current, "delta": delta}
            )
    return {
        "baseline_report": str(baseline_path),
        "threshold": threshold,
        "regressions": regressions,
        "gate": "FAIL" if regressions else "PASS",
    }


# ── Optional RAGAS multimodal soft scores (R4) ──────────────────────────────
# Report-only: these scores never enter the gate (compare_baseline reads only
# hit_rate/gold_recall/mrr plus the scope_violation error rate from the
# hard-assertion summary).


def _import_multimodal_metrics() -> tuple[Any, Any, str | None]:
    """Locate the RAGAS multimodal metric classes; never raises.

    Prefers the legacy ``ragas.metrics`` path (same decision as
    tools/evaluation_pipeline.py: the MetricWithLLM classes there accept our
    langchain ``tools.llm.get_llm`` client, while the newer
    ``ragas.metrics.collections`` variants require InstructorLLM) and falls
    back to ``ragas.metrics.collections`` for post-1.0 ragas where the legacy
    path is removed. Returns ``(faithfulness_cls, relevance_cls, skip_reason)``
    with both classes None and a reason string when neither path imports.
    """
    try:
        from ragas.metrics import MultiModalFaithfulness, MultiModalRelevance
    except Exception as exc:  # ImportError and optional-dependency failures
        first_error = f"{type(exc).__name__}: {exc}"
        try:
            from ragas.metrics.collections import (  # type: ignore[no-redef]
                MultiModalFaithfulness,
                MultiModalRelevance,
            )
        except Exception as exc2:
            return None, None, (
                f"ragas multimodal metrics not importable "
                f"(ragas.metrics: {first_error}; ragas.metrics.collections: "
                f"{type(exc2).__name__}: {exc2})"
            )
    return MultiModalFaithfulness, MultiModalRelevance, None


def _mm_metric_label(cls: Any, fallback: str) -> str:
    """ragas metric name (e.g. faithful_rate) with a stable fallback key."""
    name = str(getattr(cls, "name", "") or "").strip()
    return name or fallback


def _mm_contexts(row: dict, top_k: int) -> list[str]:
    """Retrieved contexts for one row: top-k hit text previews (non-empty)."""
    contexts: list[str] = []
    for hit in (row.get("hits") or [])[:top_k]:
        text = str(hit.get("text_preview") or "").strip()
        if text:
            contexts.append(text)
    return contexts


def _mm_metric_mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


async def apply_ragas_multimodal(
    rows: list[dict],
    results: list[dict],
    *,
    top_k: int,
) -> dict:
    """Append per-row RAGAS multimodal soft scores; return the summary section.

    ``rows`` (retrieval output, carries hit text) and ``results`` (hard
    assertion output) are parallel lists from the same run; each result row
    gets a ``ragas_multimodal`` sub-dict (metric → score, plus per-skip
    reasons). Input assembly follows the retrieval-eval shape: response =
    the evalset's gold ``reference``, user_input = question, contexts = top-k
    hit text previews.

    Degradation contract: this function never raises and never touches the
    hard metrics — metric import OR INIT failure (recorded as section-level
    ``skip_reason``), missing reference/contexts, LLM errors and NaN scores
    all become skips with reasons. Pure-text libraries may make the
    multimodal prompts error; that shows up as skips, not failures.
    """
    section: dict[str, Any] = {
        "enabled": True,
        "metric_source": None,
        "ragas_metric_names": {},
        "metrics": {
            "multimodal_faithfulness": {"mean": None, "scored": 0, "skipped": 0},
            "multimodal_relevance": {"mean": None, "scored": 0, "skipped": 0},
        },
        "skip_reason": None,
    }
    faith_cls, rel_cls, import_reason = _import_multimodal_metrics()
    if faith_cls is None or rel_cls is None:
        section["skip_reason"] = import_reason
        logger.warning("[mm-eval] RAGAS multimodal soft scores skipped: {}", import_reason)
        return section

    # Everything below (lazy imports, judge LLM construction, metric instances)
    # can fail on a degraded environment — the never-raise contract moves ALL
    # of it into the try so any failure degrades to skip_reason (review P2-6).
    try:
        from core.config import config as app_config
        from ragas.dataset_schema import SingleTurnSample
        from tools.llm import get_llm

        llm = get_llm(
            model_name=app_config.ragas_llm_model,
            temperature=0.0,
            ragas_strip_json_fence=True,
        )
        faith_metric = faith_cls(llm=llm)
        rel_metric = rel_cls(llm=llm)
    except Exception as init_exc:
        reason = f"metric init failed: {type(init_exc).__name__}: {init_exc}"[:300]
        section["skip_reason"] = reason
        logger.warning("[mm-eval] RAGAS multimodal soft scores skipped: {}", reason)
        return section

    section["metric_source"] = getattr(faith_cls, "__module__", "ragas").rsplit(".", 1)[0]
    section["ragas_metric_names"] = {
        "multimodal_faithfulness": _mm_metric_label(faith_cls, "faithful_rate"),
        "multimodal_relevance": _mm_metric_label(rel_cls, "relevance_rate"),
    }

    metric_keys = ("multimodal_faithfulness", "multimodal_relevance")
    scores: dict[str, list[float]] = {key: [] for key in metric_keys}
    counters = {key: {"scored": 0, "skipped": 0} for key in metric_keys}

    async def _score_one(metric: Any, sample: SingleTurnSample) -> float:
        value = float(await metric.single_turn_ascore(sample))
        if value != value:  # NaN (ragas returns nan when the judge yields no parse)
            raise ValueError("judge returned NaN")
        return value

    for row, result in zip(rows, results):
        per_row: dict[str, Any] = {
            "multimodal_faithfulness": None,
            "multimodal_relevance": None,
            "skip_reasons": {},
        }
        result["ragas_multimodal"] = per_row
        if row.get("error"):
            for key in metric_keys:
                counters[key]["skipped"] += 1
                per_row["skip_reasons"][key] = f"retrieval error: {row.get('error')}"
            continue
        question = row.get("question") or {}
        reference = str(question.get("reference") or "").strip()
        contexts = _mm_contexts(row, top_k)
        if not reference or not contexts:
            reason = "no_reference" if not reference else "no_nonempty_hit_contexts"
            for key in metric_keys:
                counters[key]["skipped"] += 1
                per_row["skip_reasons"][key] = reason
            continue
        sample = SingleTurnSample(
            user_input=str(question.get("question") or ""),
            response=reference,
            retrieved_contexts=contexts,
        )
        for key, metric in (
            (metric_keys[0], faith_metric),
            (metric_keys[1], rel_metric),
        ):
            try:
                value = await _score_one(metric, sample)
            except Exception as exc:
                counters[key]["skipped"] += 1
                per_row["skip_reasons"][key] = f"{type(exc).__name__}: {exc}"[:200]
                continue
            counters[key]["scored"] += 1
            scores[key].append(value)
            per_row[key] = value

    for key in metric_keys:
        section["metrics"][key] = {
            "mean": _mm_metric_mean(scores[key]),
            "scored": counters[key]["scored"],
            "skipped": counters[key]["skipped"],
        }
    return section


async def run_evaluation_core(
    *,
    evalset: Path,
    top_k: int = 8,
    keep_leaks: bool = False,
    collection: str | None = None,
    progress=None,
    ragas_multimodal: bool = False,
) -> dict:
    """Importable T2.5 step-2 core: load → retrieve → hard assertions → report dict.

    The returned dict matches the CLI report structure (generated_at / evalset /
    top_k / summary / results) but is NOT written to disk — the caller decides.
    ``collection`` routes retrieval (None = config default);
    ``progress`` receives the CLI progress lines (default: print, CLI parity).
    ``ragas_multimodal`` appends the optional RAGAS multimodal soft-score
    section (report-only; never gates and never alters hard metrics).
    """
    say = progress or print
    questions = load_evalset(evalset, keep_leaks=keep_leaks)
    say(f"evaluating {len(questions)} questions (top_k={top_k}) ...")
    rows = await retrieve_all(questions, top_k, collection=collection, progress=progress)
    results = hard_assertions(rows, top_k)
    summary = summarize(results)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evalset": str(evalset),
        "top_k": top_k,
        "summary": summary,
        "results": results,
    }
    if ragas_multimodal:
        say("ragas multimodal soft scoring ...")
        report["ragas_multimodal"] = await apply_ragas_multimodal(rows, results, top_k=top_k)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multimodal gold-set evaluation (hard assertions)")
    parser.add_argument("--evalset", type=Path, default=Path("tools/data/multimodal_evalset_draft.json"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--report-dir", type=Path, default=Path("tools/data/multimodal_eval/reports"))
    parser.add_argument("--baseline", type=Path, default=None, help="Previous report JSON for regression gate")
    parser.add_argument("--threshold", type=float, default=0.02, help="Regression threshold (default 0.02)")
    parser.add_argument("--keep-leaks", action="store_true", help="Do not auto-drop leak_suspect questions")
    parser.add_argument(
        "--ragas-multimodal",
        action="store_true",
        help=(
            "Append RAGAS multimodal faithfulness/relevance soft scores after "
            "retrieval eval (report-only; skips degrade instead of failing; "
            "never affects the gate or exit code)"
        ),
    )
    args = parser.parse_args()

    report = asyncio.run(
        run_evaluation_core(
            evalset=args.evalset,
            top_k=args.top_k,
            keep_leaks=args.keep_leaks,
            ragas_multimodal=args.ragas_multimodal,
        )
    )
    summary = report["summary"]

    if args.baseline:
        report["gate"] = compare_baseline(summary, args.baseline, args.threshold)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / f"mm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report.get("ragas_multimodal"):
        print(json.dumps(report["ragas_multimodal"], ensure_ascii=False, indent=2))
    exit_code = 0
    if args.baseline and report.get("gate"):
        gate = report["gate"]
        print("GATE:", gate["gate"], json.dumps(gate["regressions"], ensure_ascii=False))
        # Non-zero exit on FAIL so CI can block the run. Without --baseline the
        # behaviour stays exit-0 (report-only run).
        if gate["gate"] == "FAIL":
            exit_code = 1
    print(f"report: {out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
