"""Query-side structured scope extraction for retrieval narrowing.

The write side (``retrieval_fields.build_retrieval_fields``) tags every node
with ``finance_period`` / ``finance_forms`` using period regexes over the
document body. This module runs the equivalent rules over the *question*: an
explicit year or form mention ("What does the FY2012 10-K MD&A ...") becomes a
document-level hard filter so dense/sparse retrieval only competes inside the
target filings instead of across ~70 structurally identical annual reports.

Design constraints (see DEV_PROGRESS 2026-09-14 M4 analysis):
- periods/forms only — value domains are controlled (4-digit year, fixed form
  set), so a hard filter cannot misfire; sections stay soft signals (BM25 text
  enrichment + reranker already handle them).
- rule-based, no LLM: this runs in front of every retrieval call.
- granularity: only the 4-digit year is extracted (``finance_period`` is stored
  at year granularity, e.g. "2025" for a 10-Q).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Bare years in a question are almost always period qualifiers, but keep the
# write-side noise guards: "ASU 2014-15", "Item 407", "S-X 3-09" etc. must not
# narrow the scope.
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_YEAR_NOISE_PREFIX_RE = re.compile(
    r"\b(?:ASU|ASC|SAB|S-K|S-X|Topic|Item|Rule|No\.?|Note)\s*\.?\s*-?\s*(?:20\d{2})",
    re.IGNORECASE,
)

_FORM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("10-K", re.compile(r"\b10-?Ks?\b|\bannual\s+reports?\b|年报|年度报告", re.IGNORECASE)),
    ("10-Q", re.compile(r"\b10-?Qs?\b|\bquarterly\s+reports?\b|季报|季度报告", re.IGNORECASE)),
)


@dataclass
class QueryScope:
    """Structured qualifiers extracted from the question.

    ``periods``/``forms`` are hard-filter dimensions (matched against node
    ``retrieval_fields``); ``sections`` is informational only.
    """

    periods: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)

    @property
    def explicit(self) -> bool:
        return bool(self.periods or self.forms)

def _extract_periods(question: str) -> list[str]:
    noise_spans = [m.span() for m in _YEAR_NOISE_PREFIX_RE.finditer(question)]
    found: list[str] = []
    for match in _YEAR_RE.finditer(question):
        year = match.group(1)
        start = match.start()
        # skip years that are part of accounting-standard identifiers (ASU 2014-15)
        if any(ns <= start < ne for ns, ne in noise_spans):
            continue
        # a bare year in a question is already a period mention; "FY"/"财年"
        # adjacency only adds confidence, which a question rarely lacks.
        found.append(year)
    # preserve question order, dedupe
    return list(dict.fromkeys(found))

def _extract_forms(question: str) -> list[str]:
    out: list[str] = []
    for form, pattern in _FORM_PATTERNS:
        if pattern.search(question):
            out.append(form)
    return out


_SECTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("management_discussion", ("md&a", "management discussion", "管理层讨论")),
    ("risk_factors", ("risk factor", "风险因素")),
    ("liquidity", ("liquidity", "capital resources", "流动性")),
    ("business_overview", ("business overview", "业务概览")),
)


def _extract_sections(question: str) -> list[str]:
    lowered = question.lower()
    return [name for name, needles in _SECTION_RULES if any(n in lowered for n in needles)]


def resolve_query_scope(question: str) -> QueryScope:
    """Extract hard-filter dimensions from the question (pure, no I/O)."""
    text = question or ""
    return QueryScope(
        periods=_extract_periods(text),
        forms=_extract_forms(text),
        sections=_extract_sections(text),
    )
