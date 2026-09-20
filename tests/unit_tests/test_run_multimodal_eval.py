"""P1 review fixes for scripts/run_multimodal_eval.py — offline unit tests.

Covers: p95 over the filtered latency list (embed-failure rows no longer cause
IndexError), error rows counted separately and excluded from metric means, and
the gate FAIL exit code. Retrieval is faked; no Milvus/PG/network access.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import scripts.run_multimodal_eval as mm


def _ok_row(question: str, latency: float, hit: bool = True) -> dict:
    return {
        "question": {"type": "single_hop", "question": question, "gold_chunk_ids": ["c1"]},
        "hits": [{"chunk_id": "c1", "document_id": 1, "book_id": "b", "kind": "text", "score": 0.9, "page_no": 1}]
        if hit
        else [],
        "latency_ms": latency,
        "error": None,
    }


def _error_row(question: str) -> dict:
    return {
        "question": {"type": "single_hop", "question": question, "gold_chunk_ids": ["c1"]},
        "hits": [],
        "error": "embed failed",
    }


# ---------------- _p95 ----------------


def test_p95_empty_returns_none():
    assert mm._p95([]) is None


def test_p95_single_value():
    assert mm._p95([123.4]) == 123.4


def test_p95_index_uses_filtered_length():
    # 20 samples 1..20 → index int(0.95*19)=18 → value 19.0 (not an IndexError).
    values = [float(i) for i in range(1, 21)]
    assert mm._p95(values) == 19.0


# ---------------- summarize ----------------


def test_summarize_error_rows_counted_separately_excluded_from_means():
    rows = [_ok_row("q1", 100.0), _ok_row("q2", 200.0), _ok_row("q3", 300.0), _error_row("q4")]
    summary = mm.summarize([mm.hard_assertions([r], top_k=8)[0] for r in rows])
    panel = summary["single_hop"]
    assert panel["n"] == 4
    assert panel["errors"] == 1
    assert panel["hit_rate"] == 1.0  # error row's silent zero excluded
    # p95 over the 3 successful latencies: index int(0.95*2)=1 → 200.0
    assert panel["p95_latency_ms"] == 200.0
    overall = summary["overall"]
    assert overall["errors"] == 1
    assert overall["p95_latency_ms"] == 200.0


def test_summarize_all_errored_yields_none_metrics_without_crash():
    results = mm.hard_assertions([_error_row("q1"), _error_row("q2")], top_k=8)
    summary = mm.summarize(results)
    assert summary["single_hop"]["errors"] == 2
    assert summary["single_hop"]["hit_rate"] is None
    assert summary["single_hop"]["p95_latency_ms"] is None
    assert summary["overall"]["p95_latency_ms"] is None


def test_summarize_no_error_rows_keeps_errors_zero():
    results = mm.hard_assertions([_ok_row("q1", 50.0)], top_k=8)
    summary = mm.summarize(results)
    assert summary["overall"]["errors"] == 0
    assert summary["overall"]["hit_rate"] == 1.0


# ---------------- main() gate exit codes (retrieval faked) ----------------


def _write_evalset(tmp_path):
    path = tmp_path / "evalset.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"type": "single_hop", "question": "q?", "gold_chunk_ids": ["c1"], "scope": {"book_id": "b"}}
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _run_main(tmp_path, monkeypatch, baseline: dict | None, hit: bool = True) -> int:
    async def fake_retrieve(questions, top_k):
        return [_ok_row("q?", 100.0, hit=hit)]

    monkeypatch.setattr(mm, "retrieve_all", fake_retrieve)
    argv = [
        "run_multimodal_eval.py",
        "--evalset",
        str(_write_evalset(tmp_path)),
        "--report-dir",
        str(tmp_path / "reports"),
    ]
    if baseline is not None:
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        argv += ["--baseline", str(baseline_path)]
    monkeypatch.setattr(sys, "argv", argv)
    return mm.main()


def test_main_gate_fail_exits_1(tmp_path, monkeypatch):
    # Current run misses (hit_rate/mrr = 0) vs 0.99 baseline → delta < -threshold → FAIL.
    baseline = {"summary": {"overall": {"hit_rate": 0.99, "gold_recall": 0.99, "mrr": 0.99}}}
    assert _run_main(tmp_path, monkeypatch, baseline, hit=False) == 1


def test_main_gate_pass_exits_0(tmp_path, monkeypatch):
    baseline = {"summary": {"overall": {"hit_rate": 1.0, "gold_recall": 1.0, "mrr": 1.0}}}
    assert _run_main(tmp_path, monkeypatch, baseline) == 0


def test_main_without_baseline_stays_exit_0(tmp_path, monkeypatch):
    assert _run_main(tmp_path, monkeypatch, baseline=None) == 0


def test_retrieve_all_marks_embed_failure_row():
    class FakeVectorizer:
        async def embed_text(self, text):
            class Embed:
                vector = None
                error = "provider down"

            return Embed()

        async def aclose(self):
            return None

    from tools import multimodal_vectorizer

    original = multimodal_vectorizer.build_vectorizer
    multimodal_vectorizer.build_vectorizer = lambda: FakeVectorizer()
    try:
        rows = asyncio.run(
            mm.retrieve_all([{"question": "q?", "gold_chunk_ids": ["c1"]}], top_k=4)
        )
    finally:
        multimodal_vectorizer.build_vectorizer = original
    assert rows[0]["error"] == "provider down"
    assert rows[0]["hits"] == []
    assert "latency_ms" not in rows[0]


# ---------------- RAGAS multimodal soft scores (R4, flag-gated) ----------------


def _mm_row(question: str, reference: str, with_text: bool = True) -> dict:
    hits = [
        {
            "chunk_id": "c1",
            "document_id": 1,
            "book_id": "b",
            "kind": "text",
            "score": 0.9,
            "page_no": 1,
            "text_preview": "retrieved context body" if with_text else "",
        }
    ]
    return {
        "question": {
            "type": "single_hop",
            "question": question,
            "gold_chunk_ids": ["c1"],
            "reference": reference,
        },
        "hits": hits,
        "latency_ms": 100.0,
        "error": None,
    }


class _FakeJudgeLLM:
    """Stand-in for tools.llm.get_llm — never touches a provider."""


class _ScoringMetric:
    """Fake ragas metric returning a preset value (or raising)."""

    name = "fake_rate"
    error: Exception | None = None
    value: float = 1.0

    def __init__(self, llm=None):
        self.llm = llm

    async def single_turn_ascore(self, sample):
        if type(self).error is not None:
            raise type(self).error("judge exploded")
        return type(self).value


def test_apply_ragas_multimodal_import_failure_degrades():
    rows = [_mm_row("q1?", "ref1")]
    results = mm.hard_assertions(rows, top_k=4)
    monkeypatched = (None, None, "ragas multimodal metrics not importable:boom")
    original = mm._import_multimodal_metrics
    mm._import_multimodal_metrics = lambda: monkeypatched
    try:
        section = asyncio.run(mm.apply_ragas_multimodal(rows, results, top_k=4))
    finally:
        mm._import_multimodal_metrics = original
    assert section["skip_reason"] == "ragas multimodal metrics not importable:boom"
    assert section["metrics"]["multimodal_faithfulness"] == {"mean": None, "scored": 0, "skipped": 0}
    assert section["metrics"]["multimodal_relevance"] == {"mean": None, "scored": 0, "skipped": 0}
    # Early return: result rows stay untouched (no per-row sub-dict).
    assert "ragas_multimodal" not in results[0]
    # Hard metrics untouched.
    assert results[0]["metrics"]["hit"] == 1.0


def test_apply_ragas_multimodal_scoring_exception_degrades_per_row():
    rows = [_mm_row("q1?", "ref1"), _mm_row("q2?", "ref2")]
    results = mm.hard_assertions(rows, top_k=4)

    class ExplodingFaith(_ScoringMetric):
        name = "faithful_rate"
        error = RuntimeError

    class WorkingRel(_ScoringMetric):
        name = "relevance_rate"
        error = None
        value = 0.0

    monkeypatched = (ExplodingFaith, WorkingRel, None)
    original_import = mm._import_multimodal_metrics
    mm._import_multimodal_metrics = lambda: monkeypatched
    import tools.llm as tools_llm

    original_llm = tools_llm.get_llm
    tools_llm.get_llm = lambda **kwargs: _FakeJudgeLLM()
    try:
        section = asyncio.run(mm.apply_ragas_multimodal(rows, results, top_k=4))
    finally:
        mm._import_multimodal_metrics = original_import
        tools_llm.get_llm = original_llm

    faith = section["metrics"]["multimodal_faithfulness"]
    rel = section["metrics"]["multimodal_relevance"]
    assert faith["scored"] == 0 and faith["skipped"] == 2 and faith["mean"] is None
    assert rel["scored"] == 2 and rel["skipped"] == 0 and rel["mean"] == 0.0
    per_row = results[0]["ragas_multimodal"]
    assert per_row["multimodal_faithfulness"] is None
    assert "RuntimeError: judge exploded" in per_row["skip_reasons"]["multimodal_faithfulness"]
    assert per_row["multimodal_relevance"] == 0.0
    # Hard metrics untouched.
    assert all(r["metrics"]["mrr"] == 1.0 for r in results)


def test_apply_ragas_multimodal_scores_reference_against_contexts():
    rows = [
        _mm_row("q1?", "ref1"),
        _mm_row("q2?", ""),  # no reference → skipped
        _mm_row("q3?", "ref3", with_text=False),  # no contexts → skipped
        {"question": {"type": "single_hop", "question": "q4?", "gold_chunk_ids": ["c1"]}, "hits": [], "error": "embed down"},
    ]
    results = mm.hard_assertions(rows, top_k=4)

    class FaithOne(_ScoringMetric):
        name = "faithful_rate"
        value = 1.0

    class RelZero(_ScoringMetric):
        name = "relevance_rate"
        value = 0.0

    captured: dict = {}

    def fake_import():
        captured["names"] = (
            FaithOne.__name__,
            RelZero.__name__,
        )
        return FaithOne, RelZero, None

    original_import = mm._import_multimodal_metrics
    mm._import_multimodal_metrics = fake_import
    import tools.llm as tools_llm

    original_llm = tools_llm.get_llm
    tools_llm.get_llm = lambda **kwargs: _FakeJudgeLLM()
    try:
        section = asyncio.run(mm.apply_ragas_multimodal(rows, results, top_k=4))
    finally:
        mm._import_multimodal_metrics = original_import
        tools_llm.get_llm = original_llm

    faith = section["metrics"]["multimodal_faithfulness"]
    rel = section["metrics"]["multimodal_relevance"]
    assert faith == {"mean": 1.0, "scored": 1, "skipped": 3}
    assert rel == {"mean": 0.0, "scored": 1, "skipped": 3}
    assert section["ragas_metric_names"] == {
        "multimodal_faithfulness": "faithful_rate",
        "multimodal_relevance": "relevance_rate",
    }
    per_rows = results[1]["ragas_multimodal"]
    assert per_rows["skip_reasons"]["multimodal_faithfulness"] == "no_reference"
    assert results[2]["ragas_multimodal"]["skip_reasons"]["multimodal_relevance"] == "no_nonempty_hit_contexts"
    assert results[3]["ragas_multimodal"]["skip_reasons"]["multimodal_faithfulness"].startswith("retrieval error:")
    # Hard metrics untouched by soft scoring.
    assert results[0]["metrics"]["hit"] == 1.0


def test_main_ragas_flag_off_by_default(tmp_path, monkeypatch):
    async def fake_retrieve(questions, top_k):
        return [_mm_row("q?", "ref")]

    monkeypatch.setattr(mm, "retrieve_all", fake_retrieve)
    argv = [
        "run_multimodal_eval.py",
        "--evalset",
        str(_write_evalset(tmp_path)),
        "--report-dir",
        str(tmp_path / "reports"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert mm.main() == 0
    reports = list((tmp_path / "reports").glob("mm_eval_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    # Flag off (default): no ragas section, behaviour unchanged.
    assert "ragas_multimodal" not in report


def test_main_ragas_flag_on_import_failure_keeps_gate_and_exit_code(tmp_path, monkeypatch):
    async def fake_retrieve(questions, top_k):
        return [_mm_row("q?", "ref")]

    monkeypatch.setattr(mm, "retrieve_all", fake_retrieve)
    original_import = mm._import_multimodal_metrics
    mm._import_multimodal_metrics = lambda: (None, None, "ragas not installed")
    try:
        argv = [
            "run_multimodal_eval.py",
            "--evalset",
            str(_write_evalset(tmp_path)),
            "--report-dir",
            str(tmp_path / "reports"),
            "--ragas-multimodal",
            "--baseline",
            str(_write_baseline(tmp_path, hit_rate=1.0)),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        # Soft-score degradation must not flip the gate: hard metrics all hit → PASS → 0.
        assert mm.main() == 0
    finally:
        mm._import_multimodal_metrics = original_import
    reports = sorted((tmp_path / "reports").glob("mm_eval_*.json"))
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert report["ragas_multimodal"]["skip_reason"] == "ragas not installed"
    assert report["gate"]["gate"] == "PASS"


def _write_baseline(tmp_path, hit_rate: float) -> str:
    path = tmp_path / f"baseline_{hit_rate}.json"
    path.write_text(
        json.dumps({"summary": {"overall": {"hit_rate": hit_rate, "gold_recall": hit_rate, "mrr": hit_rate}}}),
        encoding="utf-8",
    )
    return str(path)


# ---------------- compare_baseline: scope_violation gate (P2-5) ----------------


def _scope_row(book_id_of_hit: str) -> dict:
    hit = {
        "chunk_id": "c1",
        "document_id": 1,
        "book_id": book_id_of_hit,
        "kind": "text",
        "score": 0.9,
        "page_no": 1,
    }
    return {
        "question": {"type": "single_hop", "question": "q?", "gold_chunk_ids": ["c1"], "scope": {"book_id": "b"}},
        "hits": [hit],
        "latency_ms": 10.0,
        "error": None,
    }


def _summary_for(rows):
    return mm.summarize(mm.hard_assertions(rows, top_k=8))


def _write_baseline_full(tmp_path, overall: dict) -> str:
    path = tmp_path / "baseline_scope.json"
    path.write_text(json.dumps({"summary": {"overall": overall}}), encoding="utf-8")
    return str(path)


def test_scope_violation_increase_fails_gate(tmp_path):
    # baseline clean (0.0), current run leaks out-of-scope hits (1.0) → FAIL,
    # even though hit_rate/gold_recall/mrr all improved (direction inverted).
    baseline = _write_baseline_full(
        tmp_path, {"hit_rate": 0.99, "gold_recall": 0.99, "mrr": 0.5, "scope_violation": 0.0}
    )
    gate = mm.compare_baseline(_summary_for([_scope_row("other")]), Path(baseline), 0.02)
    assert gate["gate"] == "FAIL"
    assert [r["metric"] for r in gate["regressions"]] == ["scope_violation"]
    regression = gate["regressions"][0]
    assert regression["previous"] == 0.0 and regression["current"] == 1.0
    assert regression["delta"] == 1.0


def test_scope_violation_within_tolerance_passes(tmp_path):
    # ±2% tolerance preserved for the error-rate direction as well.
    baseline = _write_baseline_full(
        tmp_path, {"hit_rate": 1.0, "gold_recall": 1.0, "mrr": 1.0, "scope_violation": 0.98}
    )
    gate = mm.compare_baseline(_summary_for([_scope_row("other")]), Path(baseline), 0.02)
    assert gate["gate"] == "PASS"


def test_scope_violation_improvement_never_fails(tmp_path):
    # lower-is-better: scope going DOWN must not be flagged as a regression.
    baseline = _write_baseline_full(
        tmp_path, {"hit_rate": 1.0, "gold_recall": 1.0, "mrr": 1.0, "scope_violation": 0.5}
    )
    gate = mm.compare_baseline(_summary_for([_scope_row("b")]), Path(baseline), 0.02)
    assert gate["gate"] == "PASS"


def test_legacy_baseline_without_scope_defaults_to_zero(tmp_path):
    # Legacy reports predate scope_violation: baseline treated as 0.0 so an
    # out-of-scope rate in the current run still gates (P2-5).
    baseline = _write_baseline_full(tmp_path, {"hit_rate": 1.0, "gold_recall": 1.0, "mrr": 1.0})
    gate = mm.compare_baseline(_summary_for([_scope_row("other")]), Path(baseline), 0.02)
    assert gate["gate"] == "FAIL"
    assert gate["regressions"][0]["previous"] == 0.0

    # ...and a clean current run still passes against the same legacy baseline.
    gate_ok = mm.compare_baseline(_summary_for([_scope_row("b")]), Path(baseline), 0.02)
    assert gate_ok["gate"] == "PASS"


# ---------------- apply_ragas_multimodal: init never raises (P2-6) ----------------


def test_apply_ragas_multimodal_init_failure_degrades():
    rows = [_mm_row("q1?", "ref1")]
    results = mm.hard_assertions(rows, top_k=4)

    class FaithOne(_ScoringMetric):
        name = "faithful_rate"
        value = 1.0

    class RelOne(_ScoringMetric):
        name = "relevance_rate"
        value = 1.0

    original_import = mm._import_multimodal_metrics
    mm._import_multimodal_metrics = lambda: (FaithOne, RelOne, None)
    import tools.llm as tools_llm

    def exploding_get_llm(**kwargs):
        raise RuntimeError("no judge api key")

    original_llm = tools_llm.get_llm
    tools_llm.get_llm = exploding_get_llm
    try:
        section = asyncio.run(mm.apply_ragas_multimodal(rows, results, top_k=4))
    finally:
        mm._import_multimodal_metrics = original_import
        tools_llm.get_llm = original_llm

    assert section["skip_reason"] is not None
    assert section["skip_reason"].startswith("metric init failed")
    assert "RuntimeError: no judge api key" in section["skip_reason"]
    assert section["metrics"]["multimodal_faithfulness"] == {"mean": None, "scored": 0, "skipped": 0}
    assert section["metrics"]["multimodal_relevance"] == {"mean": None, "scored": 0, "skipped": 0}
    # Hard metrics and result rows untouched.
    assert "ragas_multimodal" not in results[0]
    assert results[0]["metrics"]["hit"] == 1.0
