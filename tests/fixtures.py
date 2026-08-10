"""Håndskrevne API-formede testdata som speiler Vegvesen GraphQL-responsen."""

from typing import Dict, List, Optional


def month_entry(
    month: int,
    average: Optional[float],
    coverage: Optional[float] = None,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
) -> Dict:
    """Én rad i byMonth-listen slik API-et returnerer den."""
    ci = None
    if ci_lower is not None or ci_upper is not None:
        ci = {"lowerBound": ci_lower, "upperBound": ci_upper}
    return {
        "month": month,
        "total": {
            "volume": {"average": average, "confidenceInterval": ci},
            "coverage": {"percentage": coverage},
        },
    }


def bicycle_day(day: str, volume: Optional[float], coverage: Optional[float] = 100.0) -> Dict:
    """Én kant i byDay-listen for et sykkelpunkt."""
    return {
        "node": {
            "from": f"{day}T00:00:00+02:00",
            "total": {
                "volumeNumbers": {"volume": volume},
                "coverage": {"percentage": coverage},
            },
        }
    }


def bicycle_payload(days: List[Dict], page_info: Optional[Dict] = None) -> Dict:
    """byDay-svar for ett sykkelpunkt, valgfritt med pageInfo."""
    by_day: Dict = {"edges": days}
    if page_info is not None:
        by_day["pageInfo"] = page_info
    return {"data": {"trafficData": {"volume": {"byDay": by_day}}}}


def two_point_year(year_scale: float = 1.0) -> Dict[str, List[Dict]]:
    """To målepunkter med data for januar og februar."""
    return {
        "punkt_a": [
            month_entry(1, 1000.0 * year_scale, coverage=98.0, ci_lower=950.0, ci_upper=1050.0),
            month_entry(2, 1100.0 * year_scale, coverage=97.0, ci_lower=1040.0, ci_upper=1160.0),
        ],
        "punkt_b": [
            month_entry(1, 500.0 * year_scale, coverage=95.0, ci_lower=470.0, ci_upper=530.0),
            month_entry(2, 550.0 * year_scale, coverage=94.0, ci_lower=520.0, ci_upper=580.0),
        ],
    }
