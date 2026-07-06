"""Tester for den delte GraphQL-klienten: retry-atferd og ISO-ukeberegning."""

import pytest
import requests

from ryfast_app import vegvesen_api
from ryfast_app.vegvesen_api import VegvesenApiError, iso_week_date_range, post_graphql


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(vegvesen_api.time, "sleep", lambda _s: None)


class TestPostGraphql:
    def test_lykkes_etter_to_feil(self, monkeypatch):
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(1)
            if len(calls) < 3:
                raise requests.ConnectionError("nede")
            return _FakeResponse({"data": {"ok": True}})

        monkeypatch.setattr(vegvesen_api.requests, "post", fake_post)
        assert post_graphql("query {}", 5) == {"data": {"ok": True}}
        assert len(calls) == 3

    def test_graphql_feil_kastes_uten_retry(self, monkeypatch):
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(1)
            return _FakeResponse({"errors": [{"message": "ugyldig punkt"}]})

        monkeypatch.setattr(vegvesen_api.requests, "post", fake_post)
        with pytest.raises(VegvesenApiError, match="GraphQL error"):
            post_graphql("query {}", 5)
        assert len(calls) == 1  # deterministisk feil prøves ikke på nytt

    def test_oppbrukte_forsoek_kaster(self, monkeypatch):
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(1)
            raise requests.Timeout("timeout")

        monkeypatch.setattr(vegvesen_api.requests, "post", fake_post)
        with pytest.raises(VegvesenApiError, match="etter 3 forsøk"):
            post_graphql("query {}", 5)
        assert len(calls) == 3


class TestIsoWeekDateRange:
    def test_uke_1_2024_starter_1_januar(self):
        from_ts, to_ts = iso_week_date_range(2024, 1)
        assert from_ts == "2024-01-01T00:00:00+01:00"
        assert to_ts == "2024-01-07T23:59:59+01:00"

    def test_sommeruke_faar_sommertid_offset(self):
        from_ts, to_ts = iso_week_date_range(2024, 28)
        assert from_ts == "2024-07-08T00:00:00+02:00"
        assert to_ts == "2024-07-14T23:59:59+02:00"

    def test_uke_53_finnes_i_2020(self):
        result = iso_week_date_range(2020, 53)
        assert result is not None
        assert result[0].startswith("2020-12-28")

    def test_uke_53_finnes_ikke_i_2021(self):
        assert iso_week_date_range(2021, 53) is None

    def test_ugyldig_ukenummer(self):
        assert iso_week_date_range(2024, 60) is None
