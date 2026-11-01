"""Hent tall for veksten etter åpningen av hele Sykkelstamvegen 16. juni 2026.

2026 sammenlignes med identiske kalenderdøgn i 2025. Bare døgn med minst
50 prosent datadekning i begge år brukes, slik at datahull ikke blir til vekst.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, Optional

import pandas as pd

from ryfast_app.bicycle import parse_daily_volumes
from ryfast_app.config import (
    BICYCLE_DAILY_QUERY_TEMPLATE,
    BICYCLE_DEFAULT_OPENING_DATE,
    BICYCLE_MAX_PAGES,
    BICYCLE_MIN_COVERAGE_PCT,
    BICYCLE_POINTS,
)
from ryfast_app.vegvesen_api import OSLO_TZ, VegvesenApiError, post_graphql

BASELINE_YEAR = BICYCLE_DEFAULT_OPENING_DATE.year - 1
POINT_IDS = ("99274B3204701", "34304B3204701")


@dataclass(frozen=True)
class Comparison:
    """Nivåendring på identiske kalenderdøgn."""

    baseline_mean: float
    current_mean: float
    growth_pct: float
    days: int


def _by_day(payload: Optional[Dict]) -> Dict:
    return (
        ((payload or {}).get("data") or {})
        .get("trafficData", {})
        .get("volume", {})
        .get("byDay", {})
    ) or {}


def fetch_daily_payload(
    point_id: str,
    start_date: date,
    end_date: date,
    timeout_s: int = 30,
    max_pages: int = BICYCLE_MAX_PAGES,
) -> Dict:
    """Hent alle sider med døgndata i et inklusivt datointervall."""

    if end_date < start_date:
        raise ValueError("Sluttdato kan ikke være før startdato")
    start = datetime.combine(start_date, time.min, tzinfo=OSLO_TZ).isoformat()
    end = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=OSLO_TZ).isoformat()
    edges = []
    after: Optional[str] = None
    for _ in range(max_pages):
        after_arg = f', after: "{after}"' if after else ""
        query = BICYCLE_DAILY_QUERY_TEMPLATE.format(
            point_id=point_id, from_date=start, to_date=end, after_arg=after_arg
        )
        by_day = _by_day(post_graphql(query, timeout_s=timeout_s))
        edges.extend(by_day.get("edges") or [])
        page_info = by_day.get("pageInfo") or {}
        cursor = page_info.get("endCursor")
        if not page_info.get("hasNextPage"):
            break
        if not cursor or cursor == after:
            raise VegvesenApiError(f"Paginering for {point_id} mangler ny markør")
        after = str(cursor)
    else:
        raise VegvesenApiError(f"{point_id} oversteg sidegrensen på {max_pages} sider")
    return {"data": {"trafficData": {"volume": {"byDay": {"edges": edges}}}}}


def reliable_daily(payload: Dict, min_coverage_pct: float) -> pd.DataFrame:
    """Pakk ut pålitelige døgn og legg til kalendernøkkel på tvers av år."""

    daily = parse_daily_volumes(payload, min_coverage_pct)
    daily = daily.loc[daily["reliable"] & daily["volume"].notna()].copy()
    daily["month_day"] = daily["date"].map(lambda value: (value.month, value.day))
    return daily


def compare_period(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    start_month_day: tuple[int, int],
    end_month_day: tuple[int, int],
) -> Comparison:
    """Sammenlign bare kalenderdøgn med pålitelige tall i begge år."""

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame["month_day"].map(
            lambda value: start_month_day <= value <= end_month_day
        )
        return frame.loc[mask, ["month_day", "volume"]]

    paired = select(baseline).merge(
        select(current), on="month_day", suffixes=("_baseline", "_current")
    )
    if paired.empty:
        raise ValueError("Ingen pålitelige kalenderdøgn finnes i begge år")
    baseline_mean = float(paired["volume_baseline"].mean())
    current_mean = float(paired["volume_current"].mean())
    if not baseline_mean:
        raise ValueError("Referanseperioden har nulltrafikk")
    return Comparison(
        baseline_mean=baseline_mean,
        current_mean=current_mean,
        growth_pct=(current_mean / baseline_mean - 1) * 100,
        days=len(paired),
    )


def _point_name(point_id: str) -> str:
    return str(BICYCLE_POINTS.get(point_id, {}).get("name", point_id))


def _print_comparison(label: str, comparison: Comparison) -> None:
    print(
        f"  {label:<18} {comparison.baseline_mean:7.1f} → "
        f"{comparison.current_mean:7.1f}  "
        f"{comparison.growth_pct:+6.1f} %  ({comparison.days} døgn)"
    )


def print_report(
    frames: Dict[tuple[str, int], pd.DataFrame], through_date: date
) -> None:
    through_md = (through_date.month, through_date.day)
    print(
        f"Sykkeltrafikk 2025 → 2026, tall til og med {through_date:%d.%m.%Y}\n"
        "Snitt syklister per døgn; identiske kalenderdøgn med god dekning."
    )
    for point_id in POINT_IDS:
        baseline = frames[point_id, BASELINE_YEAR]
        current = frames[point_id, BICYCLE_DEFAULT_OPENING_DATE.year]
        before = compare_period(baseline, current, (1, 1), (6, 15))
        after = compare_period(baseline, current, (6, 16), through_md)
        print(f"\n{_point_name(point_id)} ({point_id})")
        _print_comparison("Før åpningen", before)
        _print_comparison("Etter åpningen", after)
        for label, start, end in (
            ("16.–30. juni", (6, 16), (6, 30)),
            ("Juli", (7, 1), (7, 31)),
            ("August", (8, 1), min(through_md, (8, 31))),
        ):
            if start <= through_md:
                _print_comparison(label, compare_period(baseline, current, start, end))
        post = current.loc[current["month_day"].map(lambda value: (6, 16) <= value <= through_md)]
        top = post.nlargest(3, "volume")
        top_text = ", ".join(
            f"{row.date:%d.%m}: {row.volume:.0f}" for row in top.itertuples()
        )
        print(f"  Toppdøgn 2026: {top_text}")
    print(
        "\nTolkning: Tallene dokumenterer et tidssammenfall og en år-over-år-endring, "
        "men beviser ikke alene at åpningen forårsaket veksten."
    )


def _parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Bruk datoformatet ÅÅÅÅ-MM-DD") from exc
    if parsed.year != BICYCLE_DEFAULT_OPENING_DATE.year:
        raise argparse.ArgumentTypeError("Datoen må være i 2026")
    if parsed < BICYCLE_DEFAULT_OPENING_DATE:
        raise argparse.ArgumentTypeError("Datoen kan ikke være før åpningen")
    return parsed


def main() -> int:
    last_complete_day = min(
        datetime.now(OSLO_TZ).date() - timedelta(days=1), date(2026, 12, 31)
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=_parse_date, default=last_complete_day)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--min-coverage", type=float, default=BICYCLE_MIN_COVERAGE_PCT)
    args = parser.parse_args()
    if args.through > last_complete_day:
        parser.error("Sluttdatoen kan ikke være senere enn siste komplette døgn")

    frames = {}
    try:
        for point_id in POINT_IDS:
            for year in (BASELINE_YEAR, BICYCLE_DEFAULT_OPENING_DATE.year):
                end = args.through.replace(year=year)
                payload = fetch_daily_payload(
                    point_id, date(year, 1, 1), end, args.timeout
                )
                frames[point_id, year] = reliable_daily(payload, args.min_coverage)
        print_report(frames, args.through)
    except (ValueError, VegvesenApiError) as exc:
        print(f"FEIL: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
