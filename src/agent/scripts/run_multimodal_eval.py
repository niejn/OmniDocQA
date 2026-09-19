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
3. SOFT SCORES (optional ``--ragas``): RAGAS faithfulness/context-precision on
   the retrieved contexts (existing ragas@fc0d071 + ark judge infrastructure).
4. GATE REPORT: per-metric means by question type, written next to the m4_bench
   pattern; ``--baseline`` compares against a previous report at ±threshold
   (default 2%) and flags regressions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))


def load_evalset(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") or []
    # Drop leak suspects unless explicitly kept (human curation hook).
    kept = [q for q in questions if args_keep_leaks() or not q.get("leak_suspect")]
    return kept


_KEEP_LEAKS = False


def args_keep_leaks() -> bool:
    return _KEEP_LEAKS


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


async def retrieve_all(questions: list[dict], top_k: int) -> list[dict]:
    from tools.multimodal_vectorizer import build_vectorizer
    from tools.retrieval_backends.dense_milvus_multimodal import (
        MilvusMultimodalDenseBackend,
    )

    backend = MilvusMultimodalDenseBackend()
    vectorizer = build_vectorizer()
    rows: list[dict] = []
    try:
        for i, question in enumerate(questions, 1):
            t0 = time.perf_counter()
            embed = await vectorizer.embed_text(question["question"])
            if embed.vector is None:
                rows.append({"question": question, "hits": [], "error": embed.error})
                continue
            hits = backend.search(embed.vector, limit=top_k, log_stage="mm_eval_retrieve")
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
                        }
                        for h in hits
                    ],
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "error": None,
                }
            )
            print(f"  [{i}/{len(questions)}] {len(hits)} hits in {rows[-1]['latency_ms']}ms")
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


def summarize(results: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    summary: dict[str, dict] = {}
    for qtype, items in sorted(by_type.items()):
        summary[qtype] = {
            "n": len(items),
            "hit_rate": round(statistics.mean(x["metrics"]["hit"] for x in items), 4),
            "gold_recall": round(statistics.mean(x["metrics"]["recall"] for x in items), 4),
            "mrr": round(statistics.mean(x["metrics"]["mrr"] for x in items), 4),
            "scope_violation": round(statistics.mean(x["scope_violation"] for x in items), 4),
            "p95_latency_ms": (
                round(sorted(x["latency_ms"] for x in items if x["latency_ms"])[int(0.95 * max(0, len(items) - 1))], 1)
                if any(x["latency_ms"] for x in items)
                else None
            ),
        }
    if results:
        summary["overall"] = {
            "n": len(results),
            "hit_rate": round(statistics.mean(x["metrics"]["hit"] for x in results), 4),
            "gold_recall": round(statistics.mean(x["metrics"]["recall"] for x in results), 4),
            "mrr": round(statistics.mean(x["metrics"]["mrr"] for x in results), 4),
            "scope_violation": round(statistics.mean(x["scope_violation"] for x in results), 4),
            "p95_latency_ms": None,
        }
    return summary


def compare_baseline(summary: dict, baseline_path: Path, threshold: float) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_summary = baseline.get("summary") or {}
    regressions: list[dict] = []
    for metric in ("hit_rate", "gold_recall", "mrr"):
        current = (summary.get("overall") or {}).get(metric)
        previous = (base_summary.get("overall") or {}).get(metric)
        if current is None or previous is None:
            continue
        delta = round(current - previous, 4)
        if delta < -threshold:
            regressions.append({"metric": metric, "previous": previous, "current": current, "delta": delta})
    return {
        "baseline_report": str(baseline_path),
        "threshold": threshold,
        "regressions": regressions,
        "gate": "FAIL" if regressions else "PASS",
    }


def main() -> None:
    global _KEEP_LEAKS
    parser = argparse.ArgumentParser(description="Run multimodal gold-set evaluation (hard assertions)")
    parser.add_argument("--evalset", type=Path, default=Path("tools/data/multimodal_evalset_draft.json"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--report-dir", type=Path, default=Path("tools/data/multimodal_eval/reports"))
    parser.add_argument("--baseline", type=Path, default=None, help="Previous report JSON for regression gate")
    parser.add_argument("--threshold", type=float, default=0.02, help="Regression threshold (default 0.02)")
    parser.add_argument("--keep-leaks", action="store_true", help="Do not auto-drop leak_suspect questions")
    args = parser.parse_args()
    _KEEP_LEAKS = args.keep_leaks

    questions = load_evalset(args.evalset)
    print(f"evaluating {len(questions)} questions (top_k={args.top_k}) ...")
    rows = asyncio.run(retrieve_all(questions, args.top_k))
    results = hard_assertions(rows, args.top_k)
    summary = summarize(results)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evalset": str(args.evalset),
        "top_k": args.top_k,
        "summary": summary,
        "results": results,
    }
    if args.baseline:
        report["gate"] = compare_baseline(summary, args.baseline, args.threshold)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / f"mm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.baseline and report.get("gate"):
        print("GATE:", report["gate"]["gate"], json.dumps(report["gate"]["regressions"], ensure_ascii=False))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
