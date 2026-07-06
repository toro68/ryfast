"""Streamlit-fri GraphQL-klient mot Vegvesens trafikkdata-API.

Deles mellom Streamlit-appen og CLI-verktøyet compare_vegvesen_ferde.py.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from ryfast_app.config import API_MAX_RETRIES, API_RETRY_DELAY, URL

logger = logging.getLogger(__name__)

OSLO_TZ = ZoneInfo("Europe/Oslo")


class VegvesenApiError(RuntimeError):
    """API-kallet feilet etter alle forsøk, eller svaret inneholdt GraphQL-feil."""


def post_graphql(
    query: str,
    timeout_s: int = 30,
    *,
    max_retries: int = API_MAX_RETRIES,
    retry_delay: float = API_RETRY_DELAY,
) -> Dict:
    """Kjør en GraphQL-spørring med retry og lineær backoff.

    Returnerer hele responsobjektet (inkl. toppnivå "data").
    GraphQL-feil er deterministiske og prøves ikke på nytt; nettverksfeil og
    timeout prøves inntil max_retries ganger. Kaster VegvesenApiError til slutt.
    """
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = requests.post(URL, json={"query": query}, timeout=timeout_s)
            response.raise_for_status()
            data = response.json()
            if "errors" in data:
                raise VegvesenApiError(f"GraphQL error: {data['errors']}")
            return data
        except requests.Timeout as exc:
            last_error = exc
            logger.warning("Timeout on attempt %s/%s", attempt + 1, max_retries)
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("Request failed on attempt %s/%s: %s", attempt + 1, max_retries, exc)
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))
    raise VegvesenApiError(
        f"Vegvesen API-kall feilet etter {max_retries} forsøk: {last_error}"
    ) from last_error


def iso_week_date_range(year: int, week: int) -> Optional[Tuple[str, str]]:
    """Fra/til-tidsstempler (mandag 00:00:00 til søndag 23:59:59, Europe/Oslo)
    for en ISO-uke, eller None hvis uken ikke hører til året.

    ISO 8601: 4. januar er alltid i uke 1.
    """
    jan_4 = datetime(year, 1, 4)
    week_1_monday = jan_4 - timedelta(days=jan_4.isocalendar()[2] - 1)
    week_start = week_1_monday + timedelta(weeks=week - 1)
    week_end = week_start + timedelta(days=6)
    if week_start.isocalendar()[0] != year or week_end.isocalendar()[0] != year:
        return None
    from_ts = week_start.replace(hour=0, minute=0, second=0, tzinfo=OSLO_TZ)
    to_ts = week_end.replace(hour=23, minute=59, second=59, tzinfo=OSLO_TZ)
    return from_ts.isoformat(), to_ts.isoformat()
