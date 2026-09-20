"""R2 reference-backfill script — pure logic + orchestration, fully offline.

LLM generation and retrieval are monkeypatched at the module boundary
(``_generate_reference`` / ``_retrieve_context``), so no network, no Milvus/PG,
no API keys are touched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pytest
import scripts.generate_reference_answers as gr


def _ns(tmp_path: Path, **overrides) -> argparse.Namespace:
    base = {
        "questions": None,
        "output": tmp_path / "narrative_reference_answers.json",
        "progress": tmp_path / "reference_answers_progress.json",
        "model": "test/model-x",
        "with_context": True,
        "top_k": 2,
        "limit": None,
        "concurrency": 2,
        "document_ids": None,
        "enqueue": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_questions(tmp_path: Path, n: int = 3) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "questions": [{"id": i, "question": f"Q{i}?"} for i in range(1, n + 1)],
                "document_id_range": [9801, 9802],
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------- parse_questions ----------------


def test_parse_questions_m4_style_dict():
    payload = {"questions": [{"id": 1, "question": "What did MD&A say?"}, {"id": 2, "question": "Second?"}]}
    items = gr.parse_questions(payload)
    assert items == [
        {"id": "1", "question": "What did MD&A say?"},
        {"id": "2", "question": "Second?"},
    ]


def test_parse_questions_bare_list_and_alt_field_names():
    payload = [
        {"question_text": "alt field", "qid": "a1"},
        {"query": "query field"},
        {"unrelated": "no question text"},
        "not-a-dict",
    ]
    items = gr.parse_questions(payload)
    assert items == [
        {"id": "a1", "question": "alt field"},
        {"id": "2", "question": "query field"},  # id falls back to 1-based index
    ]


def test_parse_questions_invalid_payloads():
    assert gr.parse_questions(None) == []
    assert gr.parse_questions({"no_questions_key": []}) == []
    assert gr.parse_questions(42) == []


# ---------------- extract_reference ----------------


def test_extract_reference_plain_text_passes_through():
    assert gr.extract_reference("Cash was $30.7 billion.") == "Cash was $30.7 billion."


def test_extract_reference_strips_think_blocks_and_fences():
    raw = "<think>reasoning...</think>\n```text\nRevenue grew 2%.\n```"
    assert gr.extract_reference(raw) == "Revenue grew 2%."


def test_extract_reference_json_object_pulls_reference_field():
    raw = '```json\n{"reference": "iPhone revenue was $51.3B."}\n```'
    assert gr.extract_reference(raw) == "iPhone revenue was $51.3B."


def test_extract_reference_json_object_answer_field_fallback():
    assert gr.extract_reference('{"answer": " 42% margin ", "x": 1}') == "42% margin"


def test_extract_reference_empty_and_garbage():
    assert gr.extract_reference(None) == ""
    assert gr.extract_reference("") == ""
    assert gr.extract_reference("{not json") == "{not json"


# ---------------- prompt building ----------------


def test_build_reference_prompt_with_context():
    prompt = gr.build_reference_prompt("What was FY24 revenue?", ["ctx A numbers 9.4%", "ctx B"])
    assert "What was FY24 revenue?" in prompt
    assert "ctx A numbers 9.4%" in prompt
    assert "150" in prompt  # length constraint stated
    assert "禁止编造" in prompt


def test_build_reference_prompt_question_only_mode():
    prompt = gr.build_reference_prompt("Standalone question?", None)
    assert "Standalone question?" in prompt
    assert "检索片段" not in prompt


# ---------------- checkpoint helpers ----------------


def test_progress_roundtrip(tmp_path):
    path = tmp_path / "progress.json"
    progress = {"1": {"status": "ok", "reference": "r1"}}
    gr.save_progress(path, progress)
    assert gr.load_progress(path) == progress


def test_load_progress_tolerates_missing_and_corrupt(tmp_path):
    missing = tmp_path / "nope.json"
    assert gr.load_progress(missing) == {}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{broken", encoding="utf-8")
    assert gr.load_progress(corrupt) == {}


def test_select_pending_reuses_ok_retries_failed():
    progress = {
        "1": {"status": "ok"},
        "2": {"status": "failed"},
    }
    items = [{"id": "1", "question": "a"}, {"id": "2", "question": "b"}, {"id": "3", "question": "c"}]
    assert [it["id"] for it in gr.select_pending(items, progress)] == ["2", "3"]


def test_build_output_records_shape_and_order():
    progress = {
        "2": {
            "status": "ok",
            "question": "Q2?",
            "reference": "ref2",
            "model": "m",
            "mode": "with_context",
            "generated_at": "2026-09-20T00:00:00",
        },
        "1": {"status": "failed", "reference": ""},
        "3": {"status": "ok", "reference": "   "},  # blank reference is not a record
    }
    items = [
        {"id": "3", "question": "Q3?"},
        {"id": "2", "question": "Q2?"},
        {"id": "1", "question": "Q1?"},
    ]
    records = gr.build_output_records(progress, items)
    assert records == [
        {
            "question": "Q2?",
            "reference": "ref2",
            "model": "m",
            "mode": "with_context",
            "generated_at": "2026-09-20T00:00:00",
        }
    ]


def test_parse_document_ids_ranges_and_dedupe():
    assert gr.parse_document_ids("9801,9803-9805,9801") == [9801, 9803, 9804, 9805]
    assert gr.parse_document_ids(None) == []
    assert gr.parse_document_ids("") == []


def test_document_ids_from_payload():
    assert gr.document_ids_from_payload({"document_id_range": [9801, 9803]}) == [9801, 9802, 9803]
    assert gr.document_ids_from_payload({"document_id_range": [5, 4]}) == []
    assert gr.document_ids_from_payload({}) == []
    assert gr.document_ids_from_payload("nope") == []


# ---------------- orchestration (fakes at module boundary) ----------------


def _install_fakes(monkeypatch, *, retrieve_error: Exception | None = None,
                   generate_error: Exception | None = None):
    retrieve_calls: list[str] = []
    generate_calls: list[str] = []

    async def fake_retrieve(question, document_ids, top_k):
        retrieve_calls.append(question)
        if retrieve_error:
            raise retrieve_error
        return [f"context-for:{question}"]

    async def fake_generate(question, contexts, model):
        generate_calls.append(question)
        if generate_error:
            raise generate_error
        if contexts:
            return f"ref-with-ctx:{question}"
        return f"ref-bare:{question}"

    monkeypatch.setattr(gr, "_retrieve_context", fake_retrieve)
    monkeypatch.setattr(gr, "_generate_reference", fake_generate)
    return retrieve_calls, generate_calls


def test_run_backfill_generates_checkspoint_and_output(tmp_path, monkeypatch):
    questions = _write_questions(tmp_path)
    retrieve_calls, generate_calls = _install_fakes(monkeypatch)
    args = _ns(tmp_path, questions=questions)

    exit_code = asyncio.run(gr.run_backfill(args))

    assert exit_code == 0
    # Q1 retrieved twice: once by the preflight probe, once by its worker.
    assert len(retrieve_calls) == 4 and len(generate_calls) == 3
    assert retrieve_calls[0] == "Q1?"

    records = json.loads((tmp_path / "narrative_reference_answers.json").read_text(encoding="utf-8"))
    assert [r["question"] for r in records] == ["Q1?", "Q2?", "Q3?"]
    assert all(r["reference"] == f"ref-with-ctx:Q{i}?" for i, r in zip((1, 2, 3), records))
    assert {r["model"] for r in records} == {"test/model-x"}
    assert {r["mode"] for r in records} == {"with_context"}

    progress = gr.load_progress(tmp_path / "reference_answers_progress.json")
    assert progress["1"]["status"] == "ok"


def test_run_backfill_resume_skips_completed(tmp_path, monkeypatch):
    questions = _write_questions(tmp_path, n=2)
    progress_path = tmp_path / "reference_answers_progress.json"
    gr.save_progress(
        progress_path,
        {"1": {"status": "ok", "question": "Q1?", "reference": "kept", "model": "old",
               "mode": "with_context", "generated_at": "t0"}},
    )
    _, generate_calls = _install_fakes(monkeypatch)
    args = _ns(tmp_path, questions=questions, progress=progress_path)

    exit_code = asyncio.run(gr.run_backfill(args))

    assert exit_code == 0
    assert generate_calls == ["Q2?"]  # Q1 resumed from checkpoint, not re-generated
    records = json.loads((tmp_path / "narrative_reference_answers.json").read_text(encoding="utf-8"))
    assert [r["reference"] for r in records] == ["kept", "ref-with-ctx:Q2?"]


def test_run_backfill_preflight_infra_failure_exits_2(tmp_path, monkeypatch):
    questions = _write_questions(tmp_path)
    _install_fakes(monkeypatch, retrieve_error=ConnectionError("milvus down"))
    args = _ns(tmp_path, questions=questions)

    exit_code = asyncio.run(gr.run_backfill(args))

    assert exit_code == 2
    assert not (tmp_path / "narrative_reference_answers.json").exists()


def test_run_backfill_llm_failure_recorded_exit_1(tmp_path, monkeypatch):
    questions = _write_questions(tmp_path, n=2)
    _install_fakes(monkeypatch, generate_error=ValueError("empty reference"))
    args = _ns(tmp_path, questions=questions)

    exit_code = asyncio.run(gr.run_backfill(args))

    assert exit_code == 1
    records = json.loads((tmp_path / "narrative_reference_answers.json").read_text(encoding="utf-8"))
    assert records == []  # failed questions produce no output records
    progress = gr.load_progress(tmp_path / "reference_answers_progress.json")
    assert all(rec["status"] == "failed" for rec in progress.values())


def test_run_backfill_question_only_mode_skips_retrieval(tmp_path, monkeypatch):
    questions = _write_questions(tmp_path, n=1)
    retrieve_calls, generate_calls = _install_fakes(monkeypatch)
    args = _ns(tmp_path, questions=questions, with_context=False)

    exit_code = asyncio.run(gr.run_backfill(args))

    assert exit_code == 0
    assert retrieve_calls == []
    assert generate_calls == ["Q1?"]
    records = json.loads((tmp_path / "narrative_reference_answers.json").read_text(encoding="utf-8"))
    assert records[0]["mode"] == "question_only"
    assert records[0]["reference"] == "ref-bare:Q1?"


def test_run_backfill_requires_document_ids_for_context_mode(tmp_path, monkeypatch):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [{"id": 1, "question": "Q1?"}]}), encoding="utf-8")
    _install_fakes(monkeypatch)

    exit_code = asyncio.run(gr.run_backfill(_ns(tmp_path, questions=questions)))

    assert exit_code == 2


def test_enqueue_records_pushes_reference_jobs(tmp_path, monkeypatch):
    import tools.node_repository as nr

    enqueued: list[dict] = []

    async def fake_enqueue(**kwargs):
        enqueued.append(kwargs)
        return "job-id"

    monkeypatch.setattr(nr, "enqueue_evaluation_job", fake_enqueue)

    progress = {
        "1": {
            "status": "ok",
            "question": "Q1?",
            "reference": "ref1",
            "model": "m",
            "mode": "with_context",
            "generated_at": "t0",
            "contexts": ["c1", "c2"],
        },
        "2": {"status": "failed"},
    }
    items = [{"id": "1", "question": "Q1?"}, {"id": "2", "question": "Q2?"}]

    count = asyncio.run(gr.enqueue_records(progress, items, [9801]))

    assert count == 1
    job = enqueued[0]
    assert job["query"] == "Q1?"
    assert job["reference"] == "ref1"
    assert job["context_json"] == [{"text": "c1"}, {"text": "c2"}]
    assert job["document_ids"] == [9801]
    assert job["metadata"]["source"] == "reference_backfill"


@pytest.mark.parametrize("raw,expected", [("abc\n```", "abc"), ("```json\ndef\n```", "def")])
def test_extract_reference_fence_variants(raw, expected):
    assert gr.extract_reference(raw) == expected
