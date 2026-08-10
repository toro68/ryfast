"""Tester for sykkeldata: parsing, profiler, dekning og paginering.

API-svarene bygges i minnet, så testene er nettverksfrie.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ryfast_app import api as api_mod
from ryfast_app.bicycle import (
    bicycle_point_options,
    common_period_cutoff,
    comparable_months,
    compare_years_monthly,
    coverage_summary,
    mean_points_daily,
    monthly_profile,
    panel_point_ids,
    parse_daily_volumes,
    restrict_to_common_period,
    restrict_to_comparable_months,
    retired_point_ids,
    sum_points_daily,
    weekday_profile,
    weekend_vs_weekday,
    year_comparison_summary,
    year_to_date_range,
)
from ryfast_app.config import BICYCLE_POINTS
from tests.fixtures import bicycle_day, bicycle_payload


def _uke(volumes, start="2025-06-02", coverage=100.0):
    """Én uke fra en mandag, med ett volum per døgn."""
    days = pd.date_range(start, periods=len(volumes))
    return bicycle_payload(
        [bicycle_day(d.strftime("%Y-%m-%d"), v, coverage) for d, v in zip(days, volumes)]
    )


class TestParseDailyVolumes:
    def test_kolonner_og_ukedager(self):
        # 2025-06-02 er en mandag
        out = parse_daily_volumes(_uke([100, 110, 120, 130, 140, 50, 30]))
        assert len(out) == 7
        assert out["weekday_name"].tolist() == [
            "Mandag", "Tirsdag", "Onsdag", "Torsdag", "Fredag", "Lørdag", "Søndag",
        ]
        assert out["is_weekend"].tolist() == [False] * 5 + [True] * 2
        assert out["month"].unique().tolist() == [6]
        assert out["season"].unique().tolist() == ["Sommer"]

    def test_tomt_svar_gir_tom_df_med_kolonner(self):
        for payload in (None, {}, bicycle_payload([])):
            out = parse_daily_volumes(payload)
            assert out.empty
            assert "volume" in out.columns and "reliable" in out.columns

    def test_sorteres_paa_dato(self):
        payload = bicycle_payload(
            [bicycle_day("2025-06-05", 10), bicycle_day("2025-06-02", 20)]
        )
        out = parse_daily_volumes(payload)
        assert out["date"].tolist() == [date(2025, 6, 2), date(2025, 6, 5)]

    def test_lav_dekning_beholder_volum_men_ikke_paalitelig(self):
        payload = bicycle_payload(
            [bicycle_day("2025-06-02", 100, 90.0), bicycle_day("2025-06-03", 40, 20.0)]
        )
        out = parse_daily_volumes(payload, min_coverage_pct=50.0)
        assert out["volume"].tolist() == [100.0, 40.0]
        assert out["reliable"].tolist() == [True, False]

    def test_manglende_dekning_regnes_som_upaalitelig(self):
        out = parse_daily_volumes(bicycle_payload([bicycle_day("2025-06-02", 100, None)]))
        assert out["reliable"].tolist() == [False]

    def test_manglende_volum_blir_nan(self):
        out = parse_daily_volumes(bicycle_payload([bicycle_day("2025-06-02", None)]))
        assert np.isnan(out.loc[0, "volume"])

    def test_ugyldige_rader_hoppes_over(self):
        payload = bicycle_payload(
            [
                {"node": {"from": None}},
                {"node": {"from": "ikke-en-dato"}},
                bicycle_day("2025-06-02", 100),
            ]
        )
        out = parse_daily_volumes(payload)
        assert len(out) == 1


class TestWeekdayProfile:
    def test_kalenderrekkefolge_fra_mandag(self):
        out = weekday_profile(parse_daily_volumes(_uke([100, 110, 120, 130, 140, 50, 30])))
        assert out["weekday"].tolist() == [0, 1, 2, 3, 4, 5, 6]
        assert out.iloc[0]["mean_volume"] == pytest.approx(100.0)
        assert out.iloc[6]["weekday_name"] == "Søndag"

    def test_upaalitelige_dager_utelates_som_standard(self):
        payload = bicycle_payload(
            [bicycle_day("2025-06-02", 100, 100.0), bicycle_day("2025-06-09", 10, 5.0)]
        )
        daily = parse_daily_volumes(payload)
        assert weekday_profile(daily).iloc[0]["mean_volume"] == pytest.approx(100.0)
        assert weekday_profile(daily, reliable_only=False).iloc[0]["mean_volume"] == pytest.approx(55.0)

    def test_tom_input(self):
        out = weekday_profile(pd.DataFrame())
        assert out.empty and "mean_volume" in out.columns

    def test_bare_upaalitelige_dager(self):
        daily = parse_daily_volumes(bicycle_payload([bicycle_day("2025-06-02", 10, 5.0)]))
        assert weekday_profile(daily).empty


class TestMonthlyProfile:
    def test_snitt_per_maaned(self):
        payload = bicycle_payload(
            [
                bicycle_day("2025-06-02", 100),
                bicycle_day("2025-06-03", 200),
                bicycle_day("2025-07-01", 50),
            ]
        )
        out = monthly_profile(parse_daily_volumes(payload))
        assert out["month"].tolist() == [6, 7]
        assert out.iloc[0]["mean_volume"] == pytest.approx(150.0)
        assert out.iloc[0]["days"] == 2

    def test_tom_input(self):
        assert monthly_profile(pd.DataFrame()).empty


class TestWeekendVsWeekday:
    def test_helgeandel(self):
        out = weekend_vs_weekday(parse_daily_volumes(_uke([100, 100, 100, 100, 100, 50, 30])))
        assert out["weekday_mean"] == pytest.approx(100.0)
        assert out["weekend_mean"] == pytest.approx(40.0)
        assert out["weekend_share_pct"] == pytest.approx(40.0)

    def test_uten_hverdager_gir_ingen_andel(self):
        # Bare en lørdag: andel av hverdagsnivå er ikke definert
        out = weekend_vs_weekday(parse_daily_volumes(bicycle_payload([bicycle_day("2025-06-07", 50)])))
        assert out["weekend_share_pct"] is None
        assert np.isnan(out["weekday_mean"])

    def test_tom_input(self):
        assert weekend_vs_weekday(pd.DataFrame())["weekend_share_pct"] is None


class TestCoverageSummary:
    def test_teller_dager(self):
        payload = bicycle_payload(
            [
                bicycle_day("2025-06-02", 100, 100.0),
                bicycle_day("2025-06-03", 40, 20.0),
                bicycle_day("2025-06-04", None, None),
            ]
        )
        out = coverage_summary(parse_daily_volumes(payload))
        assert out["days_total"] == 3
        assert out["days_reliable"] == 1
        assert out["days_missing"] == 1
        assert out["mean_coverage_pct"] == pytest.approx(60.0)

    def test_tom_input(self):
        out = coverage_summary(pd.DataFrame())
        assert out["days_total"] == 0 and np.isnan(out["mean_coverage_pct"])


class TestYearToDateRange:
    def test_tidligere_aar_gir_hele_aaret(self):
        assert year_to_date_range(2024, today=date(2026, 8, 10)) == (
            date(2024, 1, 1), date(2024, 12, 31)
        )

    def test_inneverende_aar_stopper_i_gaar(self):
        # I dag er ikke ferdig telt og ville sett ut som et fall i grafen
        assert year_to_date_range(2026, today=date(2026, 8, 10)) == (
            date(2026, 1, 1), date(2026, 8, 9)
        )

    def test_framtidig_aar_gir_none(self):
        assert year_to_date_range(2027, today=date(2026, 8, 10)) is None

    def test_forste_januar_gir_none(self):
        # Ingen ferdige døgn ennå i år
        assert year_to_date_range(2026, today=date(2026, 1, 1)) is None


class TestBicyclePointOptions:
    def test_nedlagte_merkes(self):
        options = bicycle_point_options(include_retired=True)
        nedlagte = [label for label in options if "nedlagt" in label]
        assert len(nedlagte) == len(retired_point_ids())
        assert nedlagte

    def test_kan_utelate_nedlagte(self):
        alle = bicycle_point_options(include_retired=True)
        i_drift = bicycle_point_options(include_retired=False)
        assert len(i_drift) == len(alle) - len(retired_point_ids())
        assert not any("nedlagt" in label for label in i_drift)

    def test_alle_punkter_har_unik_etikett(self):
        options = bicycle_point_options()
        assert len(options) == len(BICYCLE_POINTS)
        assert len(set(options.values())) == len(BICYCLE_POINTS)

    def test_sortert_paa_kommune_deretter_navn(self):
        # Noen punktnavn inneholder selv «(sykkel)», så kommunen er siste parentes
        kommuner = [
            label.rsplit("(", 1)[1].split(")")[0]
            for label in bicycle_point_options(include_retired=False)
        ]
        assert kommuner == sorted(kommuner)

    def test_punktene_har_paakrevde_felt(self):
        for pid, meta in BICYCLE_POINTS.items():
            assert meta["name"] and meta["municipality"], pid
            assert -90 <= float(meta["lat"]) <= 90, pid
            assert -180 <= float(meta["lon"]) <= 180, pid


class TestFetchBicycleDailyPagination:
    """byDay gir maks 100 døgn per side; uten paginering stopper serien stille."""

    def _fake_fetch(self, monkeypatch, pages):
        kall = []

        def _fetch(query, timeout_s, use_cache):
            kall.append(query)
            return pages[len(kall) - 1]

        monkeypatch.setattr(api_mod, "fetch_data", _fetch)
        return kall

    def test_folger_after_markoer_til_siste_side(self, monkeypatch):
        side1 = bicycle_payload(
            [bicycle_day("2025-06-02", 100)], page_info={"hasNextPage": True, "endCursor": "c1"}
        )
        side2 = bicycle_payload(
            [bicycle_day("2025-06-03", 110)], page_info={"hasNextPage": False, "endCursor": "c2"}
        )
        kall = self._fake_fetch(monkeypatch, [side1, side2])
        out = api_mod.fetch_bicycle_daily_data("p1", "f", "t", 5, False)
        assert len(kall) == 2
        assert 'after: "c1"' in kall[1] and "after:" not in kall[0]
        assert len(parse_daily_volumes(out)) == 2

    def test_stopper_uten_hasnextpage(self, monkeypatch):
        side = bicycle_payload(
            [bicycle_day("2025-06-02", 100)], page_info={"hasNextPage": False, "endCursor": "c1"}
        )
        kall = self._fake_fetch(monkeypatch, [side])
        api_mod.fetch_bicycle_daily_data("p1", "f", "t", 5, False)
        assert len(kall) == 1

    def test_uendret_markoer_stopper_loopen(self, monkeypatch):
        # Verner mot uendelig løkke hvis API-et gjentar samme endCursor
        side = bicycle_payload(
            [bicycle_day("2025-06-02", 100)], page_info={"hasNextPage": True, "endCursor": None}
        )
        kall = self._fake_fetch(monkeypatch, [side, side, side])
        api_mod.fetch_bicycle_daily_data("p1", "f", "t", 5, False)
        assert len(kall) == 1

    def test_respekterer_sidegrensen(self, monkeypatch):
        sider = [
            bicycle_payload(
                [bicycle_day("2025-06-02", 100)],
                page_info={"hasNextPage": True, "endCursor": f"c{i}"},
            )
            for i in range(10)
        ]
        kall = self._fake_fetch(monkeypatch, sider)
        api_mod.fetch_bicycle_daily_data("p1", "f", "t", 5, False, max_pages=3)
        assert len(kall) == 3

    def test_feil_paa_forste_side_gir_none(self, monkeypatch):
        self._fake_fetch(monkeypatch, [None])
        assert api_mod.fetch_bicycle_daily_data("p1", "f", "t", 5, False) is None

    def test_feil_paa_senere_side_beholder_delvis_serie(self, monkeypatch):
        side1 = bicycle_payload(
            [bicycle_day("2025-06-02", 100)], page_info={"hasNextPage": True, "endCursor": "c1"}
        )
        self._fake_fetch(monkeypatch, [side1, None])
        out = api_mod.fetch_bicycle_daily_data("p1", "f", "t", 5, False)
        assert len(parse_daily_volumes(out)) == 1

    def test_tomt_punkt_id_gir_none(self):
        assert api_mod.fetch_bicycle_daily_data("", "f", "t", 5, False) is None


class TestFetchBicycleYear:
    def test_aar_for_datastart_gir_none(self):
        assert api_mod.fetch_bicycle_year("p1", 1990, 5, False) is None

    def test_framtidig_aar_gir_none(self):
        assert api_mod.fetch_bicycle_year("p1", 2027, 5, False, today=date(2026, 8, 10)) is None

    def test_bruker_oslo_tid_og_eksklusiv_slutt(self, monkeypatch):
        fanget = {}

        def _fetch(point_id, from_date, to_date, timeout_s, use_cache):
            fanget["from"] = from_date
            fanget["to"] = to_date
            return bicycle_payload([])

        monkeypatch.setattr(api_mod, "fetch_bicycle_daily_data", _fetch)
        api_mod.fetch_bicycle_year("p1", 2026, 5, False, today=date(2026, 8, 10))
        # Sommertid i Oslo er +02:00; en hardkodet +01:00 forskjøv døgngrensen
        assert fanget["from"] == "2026-01-01T00:00:00+01:00"  # januar er normaltid
        assert fanget["to"] == "2026-08-10T00:00:00+02:00"  # dagen etter 9. august


def _punkt_df(dager, coverage=100.0, min_cov=50.0):
    """Døgn-df for ett punkt fra {dato: volum}."""
    return parse_daily_volumes(
        bicycle_payload([bicycle_day(d, v, coverage) for d, v in dager.items()]),
        min_coverage_pct=min_cov,
    )


class TestSumPointsDaily:
    def test_summerer_over_punkter(self):
        a = _punkt_df({"2025-06-02": 100, "2025-06-03": 200})
        b = _punkt_df({"2025-06-02": 50, "2025-06-03": 60})
        out = sum_points_daily({"a": a, "b": b})
        assert out["volume"].tolist() == [150.0, 260.0]
        assert out["points_present"].tolist() == [2, 2]
        assert out["points_expected"].tolist() == [2, 2]
        assert out["reliable"].all()

    def test_manglende_punkt_gir_upaalitelig_doegn(self):
        # Uten dette ville summen falle og se ut som nedgang i sykling
        a = _punkt_df({"2025-06-02": 100, "2025-06-03": 200})
        b = _punkt_df({"2025-06-02": 50})
        out = sum_points_daily({"a": a, "b": b})
        assert out["volume"].tolist() == [150.0, 200.0]
        assert out["points_present"].tolist() == [2, 1]
        assert out["reliable"].tolist() == [True, False]

    def test_lav_dekning_bidrar_ikke_til_summen(self):
        a = _punkt_df({"2025-06-02": 100}, coverage=100.0)
        b = _punkt_df({"2025-06-02": 999}, coverage=10.0)
        out = sum_points_daily({"a": a, "b": b})
        assert out["volume"].tolist() == [100.0]
        assert out["points_present"].tolist() == [1]
        assert out["reliable"].tolist() == [False]

    def test_beholder_kalenderkolonner(self):
        out = sum_points_daily({"a": _punkt_df({"2025-06-07": 40})})
        assert out.iloc[0]["weekday_name"] == "Lørdag"
        assert bool(out.iloc[0]["is_weekend"]) is True
        assert out.iloc[0]["season"] == "Sommer"

    def test_tom_input(self):
        assert sum_points_daily({}).empty
        assert sum_points_daily({"a": pd.DataFrame()}).empty

    def test_bare_lav_dekning_gir_tom(self):
        assert sum_points_daily({"a": _punkt_df({"2025-06-02": 10}, coverage=5.0)}).empty


class TestMeanPointsDaily:
    def test_snitt_per_punkt(self):
        a = _punkt_df({"2025-06-02": 100})
        b = _punkt_df({"2025-06-02": 50})
        out = mean_points_daily({"a": a, "b": b})
        assert out["volume"].tolist() == [75.0]

    def test_taaler_manglende_punkt(self):
        # Snittet skal ikke falle bare fordi ett punkt mangler
        a = _punkt_df({"2025-06-02": 100, "2025-06-03": 100})
        b = _punkt_df({"2025-06-02": 100})
        out = mean_points_daily({"a": a, "b": b})
        assert out["volume"].tolist() == [100.0, 100.0]
        assert out["reliable"].all()


class TestPanelPointIds:
    """Summen krever alle punkter til stede; panelet gjør det oppnåelig."""

    def test_utelater_punkt_uten_dekning_i_ett_aar(self):
        # Regresjon fra ekte data: med 25 punkter valgt hadde ingen døgn alle
        # punktene til stede, så hele summen ble 0 og alle nøkkeltall «N/A».
        god = {"2025-06-02": 100, "2025-06-03": 100}
        frames = {
            2025: {
                "a": _punkt_df(god),
                "b": _punkt_df(god),
                "d": _punkt_df({"2025-06-02": 10, "2025-06-03": 10}, coverage=5.0),
            },
            2026: {
                "a": _punkt_df({"2026-06-02": 120, "2026-06-03": 120}),
                "b": _punkt_df({"2026-06-02": 120, "2026-06-03": 120}),
                "d": _punkt_df({"2026-06-02": 12, "2026-06-03": 12}),
            },
        }
        # d faller ut fordi 2025 mangler dekning, selv om 2026 er fin
        assert panel_point_ids(frames) == ["a", "b"]

    def test_panelet_gir_paalitelige_doegn_i_summen(self):
        frames = {
            2025: {
                "a": _punkt_df({"2025-06-02": 100}),
                "d": _punkt_df({"2025-06-02": 10}, coverage=5.0),
            }
        }
        panel = panel_point_ids(frames)
        alle = sum_points_daily(frames[2025])
        bare_panel = sum_points_daily({p: f for p, f in frames[2025].items() if p in panel})
        assert int(alle["reliable"].sum()) == 0  # d mangler, så ingenting er pålitelig
        assert int(bare_panel["reliable"].sum()) == 1

    def test_punkt_med_kort_serie_faller_ut(self):
        # Terskelen måles mot det punktet som dekker perioden bredest, ellers
        # ville et punkt med bare én dag bestått med full score.
        frames = {
            2025: {
                "a": _punkt_df({f"2025-06-{d:02d}": 100 for d in range(1, 21)}),
                "kort": _punkt_df({"2025-06-01": 100}),
            }
        }
        assert panel_point_ids(frames) == ["a"]

    def test_tom_input(self):
        assert panel_point_ids({}) == []
        assert panel_point_ids({2025: {}}) == []


class TestCommonPeriod:
    def test_kutter_til_korteste_aar(self):
        i_fjor = _punkt_df({"2024-01-01": 10, "2024-12-31": 20})
        i_aar = _punkt_df({"2025-01-01": 30, "2025-06-30": 40})
        out = restrict_to_common_period({2024: i_fjor, 2025: i_aar})
        assert common_period_cutoff({2024: i_fjor, 2025: i_aar}) == (6, 30)
        assert len(out[2024]) == 1  # 31.12 kuttet bort
        assert len(out[2025]) == 2

    def test_uavhengig_av_sortering(self):
        usortert = parse_daily_volumes(
            bicycle_payload([bicycle_day("2025-06-30", 40), bicycle_day("2025-01-01", 30)])
        )
        assert common_period_cutoff({2025: usortert}) == (6, 30)

    def test_tom_input(self):
        assert common_period_cutoff({}) is None
        assert restrict_to_common_period({}) == {}


class TestYearComparisonSummary:
    def test_endring_mot_eldste_aar(self):
        i_fjor = _punkt_df({"2024-06-02": 100, "2024-06-03": 100})
        i_aar = _punkt_df({"2025-06-02": 150, "2025-06-03": 150})
        out = year_comparison_summary({2024: i_fjor, 2025: i_aar})
        assert out["year"].tolist() == [2024, 2025]
        # NaN, ikke None: å sette None i en float-kolonne gir NaN
        assert pd.isna(out.iloc[0]["change_pct"])
        assert out.iloc[1]["change_pct"] == pytest.approx(50.0)
        assert out.iloc[1]["total_volume"] == pytest.approx(300.0)

    def test_referanseaar_uten_nivaa_gir_ingen_endring(self):
        null = _punkt_df({"2024-06-02": 0})
        i_aar = _punkt_df({"2025-06-02": 100})
        out = year_comparison_summary({2024: null, 2025: i_aar})
        assert out["change_pct"].isna().all()

    def test_tom_input(self):
        assert year_comparison_summary({}).empty


class TestCompareYearsMonthly:
    def test_lang_form_per_aar_og_maaned(self):
        i_fjor = _punkt_df({"2024-06-02": 100, "2024-07-02": 200})
        i_aar = _punkt_df({"2025-06-02": 150})
        out = compare_years_monthly({2024: i_fjor, 2025: i_aar})
        assert set(out.columns) == {"year", "month", "mean_volume", "days"}
        assert out[out["year"] == 2024]["month"].tolist() == [6, 7]
        assert out[out["year"] == 2025]["mean_volume"].tolist() == [150.0]

    def test_tom_input(self):
        assert compare_years_monthly({}).empty


class TestComparableMonths:
    def test_bare_felles_maaneder(self):
        # 2024 mangler januar; da skal januar ut av sammenligningen
        i_fjor = _punkt_df({"2024-06-02": 100, "2024-07-02": 100})
        i_aar = _punkt_df({"2025-01-02": 20, "2025-06-02": 150, "2025-07-02": 150})
        assert comparable_months({2024: i_fjor, 2025: i_aar}) == [6, 7]

    def test_upaalitelige_maaneder_teller_ikke(self):
        i_fjor = _punkt_df({"2024-06-02": 100})
        # Juli finnes i 2025, men bare med lav dekning
        i_aar = parse_daily_volumes(
            bicycle_payload(
                [bicycle_day("2025-06-02", 150, 100.0), bicycle_day("2025-07-02", 150, 5.0)]
            )
        )
        assert comparable_months({2024: i_fjor, 2025: i_aar}) == [6]

    def test_ingen_overlapp_gir_tom(self):
        assert comparable_months(
            {2024: _punkt_df({"2024-06-02": 100}), 2025: _punkt_df({"2025-01-02": 100})}
        ) == []

    def test_tom_input(self):
        assert comparable_months({}) == []
        assert restrict_to_comparable_months({}) == {}

    def test_fjerner_dekningsskjevhet(self):
        # Regresjon fra ekte data: 2024 hadde bare vår/sommer med god dekning,
        # 2026 hadde hele året. Uten månedsfiltrering ga det «-52 %», som bare
        # var sommer målt mot helår.
        i_fjor = _punkt_df({"2024-06-02": 400, "2024-06-03": 400})
        i_aar = _punkt_df(
            {"2025-01-02": 50, "2025-01-03": 50, "2025-06-02": 400, "2025-06-03": 400}
        )
        raa = year_comparison_summary({2024: i_fjor, 2025: i_aar})
        assert raa.iloc[1]["change_pct"] < -40  # artefaktet

        filtrert = restrict_to_comparable_months({2024: i_fjor, 2025: i_aar})
        out = year_comparison_summary(filtrert)
        assert out.iloc[1]["change_pct"] == pytest.approx(0.0)  # ingen reell endring
