"""P0-1 regression: the delete CLI must pass argparse's dest (document_id) to main_async.

The old code read ``args.document_ids`` while ``--document-id`` stores to
``document_id`` — every CLI invocation crashed with AttributeError before
touching PG/Milvus. main_async is faked so the REAL argparse surface and
main() wiring run offline (no PG, no Milvus).
"""

from __future__ import annotations

import sys

import pytest
import scripts.delete_ingested_document as cli


@pytest.mark.parametrize(
    ("argv_extra", "expected_skip_pg"),
    [([], False), (["--skip-pg"], True)],
)
def test_main_passes_parsed_ids_to_main_async(monkeypatch, argv_extra, expected_skip_pg) -> None:
    captured: dict = {}

    async def fake_main_async(document_ids, *, skip_pg):
        captured["document_ids"] = document_ids
        captured["skip_pg"] = skip_pg
        return {"success": True, "results": []}

    monkeypatch.setattr(cli, "main_async", fake_main_async)
    monkeypatch.setattr(
        sys,
        "argv",
        ["delete_ingested_document.py", "--document-id", "999", "1000", *argv_extra],
    )
    cli.main()  # must not raise AttributeError: args.document_ids
    assert captured["document_ids"] == [999, 1000]
    assert captured["skip_pg"] is expected_skip_pg
