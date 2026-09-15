"""Unit tests for the query-side scope resolver (M4 context_precision fix)."""

from __future__ import annotations

from tools.finance.query_scope_resolver import resolve_query_scope


def test_fy_prefixed_year_with_form() -> None:
    scope = resolve_query_scope("What does the FY2012 10-K MD&A explain about net sales?")
    assert scope.periods == ["2012"]
    assert scope.forms == ["10-K"]
    assert scope.sections == ["management_discussion"]
    assert scope.explicit


def test_space_separated_fiscal_year() -> None:
    scope = resolve_query_scope("Why did net sales increase in fiscal 2024?")
    assert scope.periods == ["2024"]
    assert scope.explicit


def test_multiple_years_preserve_order() -> None:
    scope = resolve_query_scope("Compare risk factors in 2024 and 2023 10-Ks")
    assert scope.periods == ["2024", "2023"]
    assert scope.forms == ["10-K"]


def test_period_end_date_yields_year() -> None:
    scope = resolve_query_scope(
        "What risk factors does the filing covering period-end September 26, 2020 disclose?"
    )
    assert scope.periods == ["2020"]


def test_chinese_fiscal_year_and_form() -> None:
    scope = resolve_query_scope("2024 财年年报中 iPhone 收入为何增长？")
    assert scope.periods == ["2024"]
    assert scope.forms == ["10-K"]  # 年报


def test_no_signal_is_not_explicit() -> None:
    scope = resolve_query_scope("iPhone net sales increase")
    assert not scope.explicit
    assert scope.periods == []
    assert scope.forms == []


def test_accounting_standard_noise_not_a_period() -> None:
    scope = resolve_query_scope("How does ASU 2014-15 affect revenue recognition?")
    assert scope.periods == []
    assert not scope.explicit


def test_numbers_glued_to_year_ignored() -> None:
    # long digit runs (ids, prices) must not leak a year
    scope = resolve_query_scope("What is document 120245 about?")
    assert scope.periods == []


def test_form_only_scope_is_explicit() -> None:
    scope = resolve_query_scope("Summarize the annual report risk factors")
    assert scope.forms == ["10-K"]
    assert scope.periods == []
    assert scope.explicit


def test_sections_are_informational_only() -> None:
    # sections never make the scope explicit (soft signal by design)
    scope = resolve_query_scope("What are the liquidity considerations?")
    assert scope.sections == ["liquidity"]
    assert not scope.explicit
