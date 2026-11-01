"""Nettverksfrie tester av vekstscriptet for Sykkelstamvegen."""

from datetime import date

import pytest

import verify_bicycle_growth as growth
from tests.fixtures import bicycle_day, bicycle_payload


def test_fetch_daily_payload_paginerer(monkeypatch):
    pages = [
        bicycle_payload(
            [bicycle_day("2026-06-16", 100)],
            page_info={"hasNextPage": True, "endCursor": "neste"},
        ),
        bicycle_payload(
            [bicycle_day("2026-06-17", 200)],
            page_info={"hasNextPage": False, "endCursor": "slutt"},
        ),
    ]
    queries = []

    def fake_post(query, timeout_s):
        queries.append((query, timeout_s))
        return pages.pop(0)

    monkeypatch.setattr(growth, "post_graphql", fake_post)
    payload = growth.fetch_daily_payload(
        "punkt", date(2026, 6, 16), date(2026, 6, 17), timeout_s=7
    )
    assert len(growth._by_day(payload)["edges"]) == 2
    assert "2026-06-18T00:00:00+02:00" in queries[0][0]
    assert 'after: "neste"' in queries[1][0]
    assert queries[0][1] == 7


def test_compare_period_bruker_bare_identiske_paalitelige_dager():
    baseline = growth.reliable_daily(
        bicycle_payload(
            [
                bicycle_day("2025-06-16", 100),
                bicycle_day("2025-06-17", 200),
                bicycle_day("2025-06-18", 9_999, coverage=10),
            ]
        ),
        50,
    )
    current = growth.reliable_daily(
        bicycle_payload(
            [
                bicycle_day("2026-06-16", 150),
                bicycle_day("2026-06-17", 300),
                bicycle_day("2026-06-18", 400),
            ]
        ),
        50,
    )
    result = growth.compare_period(baseline, current, (6, 16), (6, 18))
    assert result.days == 2
    assert result.baseline_mean == 150
    assert result.current_mean == 225
    assert result.growth_pct == pytest.approx(50)


def test_aapningsdato_er_16_juni_2026():
    assert growth.BICYCLE_DEFAULT_OPENING_DATE == date(2026, 6, 16)
    assert growth.BASELINE_YEAR == 2025
    assert growth._parse_date("2026-08-30") == date(2026, 8, 30)
