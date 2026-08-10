"""Karakteriseringstester for radnivå-databehandling."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ryfast_app.processing import (
    add_month_names,
    assessable_months,
    days_in_year,
    detect_monthly_anomalies,
    format_number,
    overlapping_months,
    sum_traffic_data,
    sum_weekly_traffic_data,
)
from tests.fixtures import month_entry, two_point_year


class TestFormatNumber:
    def test_nan_gir_na(self):
        assert format_number(np.nan) == "N/A"

    def test_heltall_med_mellomrom(self):
        assert format_number(1234567) == "1 234 567"

    def test_flyttall_med_en_desimal(self):
        # Norsk format: mellomrom som tusenskiller, komma som desimaltegn
        assert format_number(1234.5) == "1 234,5"

    def test_heltallsverdi_som_float(self):
        assert format_number(1000.0) == "1 000"

    def test_numerisk_streng(self):
        assert format_number("1000") == "1 000"

    def test_ikke_numerisk_streng_uendret(self):
        assert format_number("abc") == "abc"


class TestDaysInYear:
    def test_skuddaar(self):
        assert days_in_year(2024) == 366

    def test_vanlig_aar(self):
        assert days_in_year(2023) == 365


class TestSumTrafficData:
    def test_summerer_per_maaned(self):
        sums, ci, has_data, present = sum_traffic_data(two_point_year())
        assert sums[0] == pytest.approx(1500.0)
        assert sums[1] == pytest.approx(1650.0)
        assert sums[2] == 0.0
        assert ci[0] == {"lower": pytest.approx(1420.0), "upper": pytest.approx(1580.0)}
        assert has_data[:3] == [True, True, False]
        assert present[:3] == [2, 2, 0]

    def test_tom_input(self):
        sums, ci, has_data, present = sum_traffic_data({})
        assert sums == [0.0] * 12
        assert has_data == [False] * 12
        assert present == [0] * 12

    def test_hopper_over_manglende_volum(self):
        data = {"p": [month_entry(1, None)]}
        sums, _, has_data, present = sum_traffic_data(data)
        assert sums[0] == 0.0
        assert has_data[0] is False
        assert present[0] == 0

    def test_estimate_missing_points_skalerer(self):
        # 2 av 4 forventede punkter til stede -> verdier skaleres med 2
        sums, ci, _, present = sum_traffic_data(
            two_point_year(),
            expected_point_ids=["a", "b", "c", "d"],
            estimate_missing_points=True,
        )
        assert present[0] == 2
        assert sums[0] == pytest.approx(3000.0)
        assert ci[0]["lower"] == pytest.approx(2840.0)
        assert ci[0]["upper"] == pytest.approx(3160.0)

    def test_ingen_skalering_uten_flagg(self):
        sums, _, _, _ = sum_traffic_data(
            two_point_year(),
            expected_point_ids=["a", "b", "c", "d"],
            estimate_missing_points=False,
        )
        assert sums[0] == pytest.approx(1500.0)


class TestSumWeeklyTrafficData:
    def test_summerer_per_uke(self):
        weekly = {"Uke 1": {"a": 100.0, "b": 50.0}, "Uke 2": {"a": 200.0}}
        assert sum_weekly_traffic_data(weekly) == {"Uke 1": 150.0, "Uke 2": 200.0}

    def test_tom_input(self):
        assert sum_weekly_traffic_data({}) == {}


class TestAddMonthNames:
    def test_legger_til_navn_og_kolonnerekkefolge(self):
        df = pd.DataFrame({"Month": [1, 2, 12], "2024": [10, 20, 30]})
        out = add_month_names(df)
        assert list(out.columns) == ["Month", "Month Name", "2024"]
        assert list(out["Month Name"]) == ["Januar", "Februar", "Desember"]

    def test_uten_month_kolonne_uendret(self):
        df = pd.DataFrame({"x": [1]})
        out = add_month_names(df)
        assert list(out.columns) == ["x"]


class TestAssessableMonths:
    def test_tidligere_aar_gir_alle_maaneder(self):
        assert assessable_months(2024, today=date(2026, 8, 10)) == list(range(1, 13))

    def test_innevaerende_aar_utelater_paagaaende_maaned(self):
        # August er ikke ferdig, så bare jan-jul kan vurderes
        assert assessable_months(2026, today=date(2026, 8, 10)) == list(range(1, 8))

    def test_fremtidig_aar_gir_ingen_maaneder(self):
        assert assessable_months(2027, today=date(2026, 8, 10)) == []

    def test_januar_i_innevaerende_aar_gir_ingen_maaneder(self):
        assert assessable_months(2026, today=date(2026, 1, 15)) == []


class TestOverlappingMonths:
    def test_bare_maaneder_der_begge_aar_har_tall(self):
        df = pd.DataFrame(
            {
                "Month": list(range(1, 13)),
                "2025": pd.array([100] * 12, dtype="Int64"),
                "2026": pd.array([110] * 7 + [pd.NA] * 5, dtype="Int64"),
            }
        )
        assert overlapping_months(df, ["2025", "2026"]) == list(range(1, 8))

    def test_manglende_kolonne_gir_tom_liste(self):
        df = pd.DataFrame({"Month": [1], "2025": [100.0]})
        assert overlapping_months(df, ["2025", "2026"]) == []

    def test_tom_df(self):
        assert overlapping_months(pd.DataFrame(), ["2025"]) == []


class TestDetectMonthlyAnomalies:
    def test_finner_avvik_over_terskel(self):
        df = pd.DataFrame(
            {
                "Month": [1, 2],
                "2023": [100.0, 100.0],
                "2024": [130.0, 105.0],
            }
        )
        out = detect_monthly_anomalies(df, threshold_pct=20.0)
        # Med to år vurderes bare det seneste, ellers ville januar flagges to
        # ganger med motsatt fortegn (+30% og -23%) for samme observasjon.
        assert set(out["month"]) == {1}
        assert set(out["year"]) == {2024}
        row_2024 = out[out["year"] == 2024].iloc[0]
        assert row_2024["deviation_pct"] == pytest.approx(30.0)
        assert row_2024["month_name"] == "Januar"

    def test_tre_aar_bruker_median_av_ovrige(self):
        # Med >=2 andre år er medianen et reelt forventningsnivå, og alle år vurderes
        df = pd.DataFrame({"Month": [1], "2023": [100.0], "2024": [100.0], "2025": [200.0]})
        out = detect_monthly_anomalies(df, threshold_pct=20.0)
        assert 2025 in set(out["year"])
        rad = out[out["year"] == 2025].iloc[0]
        assert rad["expected"] == pytest.approx(100.0)
        assert rad["deviation_pct"] == pytest.approx(100.0)

    def test_ingen_avvik_under_terskel(self):
        df = pd.DataFrame({"Month": [1], "2023": [100.0], "2024": [110.0]})
        out = detect_monthly_anomalies(df, threshold_pct=20.0)
        assert out.empty

    def test_krever_minst_to_aar(self):
        df = pd.DataFrame({"Month": [1], "2024": [100.0]})
        assert detect_monthly_anomalies(df).empty

    def test_tom_df(self):
        assert detect_monthly_anomalies(pd.DataFrame()).empty
