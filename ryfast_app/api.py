"""Streamlit-laget rundt Vegvesen-API-et: caching, feilbuffer og parallell henting."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from threading import Lock
from typing import Dict, List, Optional, Tuple

import streamlit as st

from ryfast_app.config import (
    API_CACHE_TTL,
    API_ERROR_BUFFER_MAX,
    API_ERROR_SESSION_MAX,
    BICYCLE_DAILY_QUERY_TEMPLATE,
    BICYCLE_DATA_START_YEAR,
    BICYCLE_MAX_PAGES,
    DATA_START_YEAR,
    MAX_BATCH_WORKERS,
    MAX_WEEKLY_WORKERS,
    QUERY_TEMPLATE,
    WEEKLY_QUERY_TEMPLATE,
)
from ryfast_app.bicycle import year_to_date_range
from ryfast_app.vegvesen_api import OSLO_TZ, VegvesenApiError, iso_week_date_range, post_graphql

logger = logging.getLogger(__name__)

# Trådsikker buffer for API-feil: worker-tråder kan ikke skrive til
# st.session_state, så feil samles her og flushes fra hovedtråden.
API_ERROR_BUFFER_LOCK = Lock()
API_ERROR_BUFFER: List[Dict[str, Optional[str]]] = []


def _build_api_error_entry(message: str, query: Optional[str] = None) -> Dict[str, Optional[str]]:
    return {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": str(message),
        "query": (query[:600] + "…") if query and len(query) > 600 else query,
    }


def flush_api_error_buffer_to_session_state() -> None:
    pending: List[Dict[str, Optional[str]]] = []
    with API_ERROR_BUFFER_LOCK:
        if not API_ERROR_BUFFER:
            return
        pending = API_ERROR_BUFFER.copy()
        API_ERROR_BUFFER.clear()

    try:
        errors = list(st.session_state.get("api_errors", []))
        errors.extend(pending)
        st.session_state.api_errors = errors[-API_ERROR_SESSION_MAX:]
    except Exception as exc:
        with API_ERROR_BUFFER_LOCK:
            API_ERROR_BUFFER[:0] = pending
            del API_ERROR_BUFFER[:-API_ERROR_BUFFER_MAX]
        logger.debug("flush_api_error_buffer: klarte ikke oppdatere session_state: %s", exc)


def record_api_error(message: str, query: Optional[str] = None) -> None:
    entry = _build_api_error_entry(message, query=query)
    with API_ERROR_BUFFER_LOCK:
        API_ERROR_BUFFER.append(entry)
        del API_ERROR_BUFFER[:-API_ERROR_BUFFER_MAX]


def clear_api_errors() -> None:
    with API_ERROR_BUFFER_LOCK:
        API_ERROR_BUFFER.clear()
    st.session_state.api_errors = []


def _fetch_data_uncached(query: str, timeout_s: int) -> Optional[Dict]:
    try:
        return post_graphql(query, timeout_s)
    except VegvesenApiError as exc:
        logger.error("API-kall feilet: %s", exc)
        record_api_error(str(exc), query=query)
        return None


@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def _fetch_data_cached(query: str, timeout_s: int) -> Optional[Dict]:
    return _fetch_data_uncached(query, timeout_s)


def fetch_data(query: str, timeout_s: int, use_cache: bool) -> Optional[Dict]:
    return _fetch_data_cached(query, timeout_s) if use_cache else _fetch_data_uncached(query, timeout_s)


def fetch_batch_traffic_data(point_ids: List[str], year: int, timeout_s: int, use_cache: bool) -> Dict[str, List[Dict]]:
    if year < DATA_START_YEAR or not point_ids:
        return {}

    fetch_fn = _fetch_data_cached if use_cache else _fetch_data_uncached
    result: Dict[str, List[Dict]] = {}
    with ThreadPoolExecutor(max_workers=min(len(point_ids), MAX_BATCH_WORKERS)) as executor:
        future_to_point = {
            executor.submit(fetch_fn, QUERY_TEMPLATE.format(point_id=pid, year=year), timeout_s): pid
            for pid in point_ids
        }
        for future in as_completed(future_to_point):
            point_id = future_to_point[future]
            try:
                data = future.result()
                if data and data.get("data", {}).get("trafficData"):
                    monthly = data["data"]["trafficData"]["volume"]["average"]["daily"]["byMonth"]
                    if monthly:
                        result[point_id] = monthly
            except (KeyError, TypeError, ValueError) as e:
                logger.error("Feil ved henting av data for punkt %s, år %s: %s", point_id, year, str(e))
                record_api_error(f"Punkt {point_id} feil (år {year}): {e}")

    return result


def fetch_weekly_traffic_data(
    point_ids: List[str], year: int, week_numbers: List[int], timeout_s: int, use_cache: bool
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    if year < DATA_START_YEAR:
        return {}, {}

    valid_weeks: List[Tuple[int, str, str]] = []
    for week_num in week_numbers:
        date_range = iso_week_date_range(year, week_num)
        if date_range is None:
            continue
        valid_weeks.append((week_num, date_range[0], date_range[1]))

    if not valid_weeks or not point_ids:
        return {}, {}

    result: Dict[str, Dict[str, float]] = {}
    cov_result: Dict[str, Dict[str, float]] = {}
    fetch_fn = _fetch_data_cached if use_cache else _fetch_data_uncached

    def _fetch_one(week_num: int, point_id: str, from_date: str, to_date: str):
        query = WEEKLY_QUERY_TEMPLATE.format(point_id=point_id, from_date=from_date, to_date=to_date)
        return week_num, point_id, fetch_fn(query, timeout_s)

    max_workers = min(len(valid_weeks) * len(point_ids), MAX_WEEKLY_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one, week_num, point_id, from_date, to_date): (week_num, point_id)
            for week_num, from_date, to_date in valid_weeks
            for point_id in point_ids
        }
        for future in as_completed(futures):
            try:
                week_num, point_id, data = future.result()
                if data and data.get("data", {}).get("trafficData"):
                    edges = data["data"]["trafficData"]["volume"]["byDay"]["edges"] or []
                    total_volume = 0.0
                    valid_days = 0
                    cov_sum = 0.0
                    cov_days = 0
                    for edge in edges:
                        volume = edge["node"]["total"]["volumeNumbers"]["volume"]
                        if volume is not None:
                            total_volume += float(volume)
                            valid_days += 1
                        cov = edge["node"]["total"]["coverage"]["percentage"]
                        if cov is not None:
                            cov_sum += float(cov)
                            cov_days += 1
                    week_key = f"Uke {week_num}"
                    if valid_days:
                        result.setdefault(week_key, {})[point_id] = total_volume / valid_days
                    if cov_days:
                        cov_result.setdefault(week_key, {})[point_id] = cov_sum / cov_days
            except (KeyError, TypeError, ValueError) as e:
                wn, pid = futures[future]
                logger.error("Feil ved henting av ukesdata for uke %s, punkt %s: %s", wn, pid, str(e))
    return result, cov_result


def _bicycle_by_day(payload: Optional[Dict]) -> Dict:
    """Plukk ut byDay-noden, eller en tom node hvis svaret mangler den."""
    return (
        ((payload or {}).get("data") or {}).get("trafficData", {}).get("volume", {}).get("byDay")
    ) or {}


def fetch_bicycle_daily_data(
    point_id: str,
    from_date: str,
    to_date: str,
    timeout_s: int,
    use_cache: bool,
    max_pages: int = BICYCLE_MAX_PAGES,
) -> Optional[Dict]:
    """Hent døgnvolum for ett sykkelpunkt i et datointervall.

    byDay gir maks 100 døgn per side, så et år må hentes over flere sider.
    Sidene slås sammen til én payload med samme struktur som et enkeltsvar,
    slik at `ryfast_app.bicycle.parse_daily_volumes` ikke trenger å kjenne
    pagineringen. Returnerer None ved feil på første side.

    Ett punkt per kall: sykkelvisningen viser ett punkt om gangen, så det er
    ikke noe å parallellisere på punktnivå. Sidene må hentes i rekkefølge
    fordi hver markør kommer fra forrige svar.
    """
    if not point_id:
        return None

    edges: List[Dict] = []
    after: Optional[str] = None
    for _ in range(max_pages):
        after_arg = f', after: "{after}"' if after else ""
        query = BICYCLE_DAILY_QUERY_TEMPLATE.format(
            point_id=point_id, from_date=from_date, to_date=to_date, after_arg=after_arg
        )
        payload = fetch_data(query, timeout_s, use_cache)
        by_day = _bicycle_by_day(payload)
        page_edges = by_day.get("edges") or []
        edges.extend(page_edges)

        if payload is None:
            if not edges:
                # Første side feilet: skill mellom feil og et punkt uten data.
                return None
            # Senere side feilet: behold det vi har, men ikke la det passere
            # stille — grafen vil se avkortet ut uten at noe er galt med punktet.
            logger.warning(
                "Sykkeldata for punkt %s: en side feilet etter %s døgn; viser delvis serie.",
                point_id,
                len(edges),
            )
            break

        page_info = by_day.get("pageInfo") or {}
        next_cursor = page_info.get("endCursor")
        # Uten ny markør ville neste runde hentet samme side om igjen.
        if not page_info.get("hasNextPage") or not next_cursor or next_cursor == after:
            break
        after = next_cursor
    else:
        logger.warning(
            "Sykkeldata for punkt %s nådde sidegrensen (%s sider); resten er utelatt.",
            point_id,
            max_pages,
        )

    return {"data": {"trafficData": {"volume": {"byDay": {"edges": edges}}}}}


def fetch_bicycle_year(
    point_id: str,
    year: int,
    timeout_s: int,
    use_cache: bool,
    today: Optional[date] = None,
) -> Optional[Dict]:
    """Hent ett år med døgndata, avkortet mot dagens dato.

    Tidsstemplene bruker Europe/Oslo via ZoneInfo framfor en fast offset:
    med hardkodet +01:00 forskyves døgngrensen en time gjennom sommertiden,
    og siste døgn faller på feil side av det eksklusive `to`.
    """
    if year < BICYCLE_DATA_START_YEAR:
        return None
    span = year_to_date_range(year, today=today)
    if span is None:
        return None
    start, end = span
    from_str = datetime.combine(start, time(0, 0), tzinfo=OSLO_TZ).isoformat()
    # `to` er eksklusiv i byDay, så vi ber om midnatt dagen etter sluttdatoen.
    to_str = datetime.combine(end + timedelta(days=1), time(0, 0), tzinfo=OSLO_TZ).isoformat()
    return fetch_bicycle_daily_data(point_id, from_str, to_str, timeout_s, use_cache)
