"""Karakteriseringstester for dekningssammendrag, totaler, vekst og sesongmønstre."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ryfast_app.metrics import (
    calculate_growth_rates,
    calculate_seasonal_patterns,
    compute_monthly_coverage_summary,
    compute_monthly_totals_table,
    coverage_pivot,
    totals_with_uncertainty_from_metrics,
)
from ryfast_app.processing import (
    calculate_yearly_total_from_monthly_averages,
    extract_point_monthly_metrics,
)
from tests.fixtures import two_point_year


class TestCalculateYearlyTotal:
    def test_en_maaned(self):
        df = pd.DataFrame({"Month": [1], "2024": [1000.0]})
        total, months, days = calculate_yearly_total_from_monthly_averages(df, 2024)
        assert total == pytest.approx(31000.0)
        assert months == 1
        assert days == 31

    def test_februar_skuddaar(self):
        df = pd.DataFrame({"Month": [2], "2024": [1000.0]})
        total, months, days = calculate_yearly_total_from_monthly_averages(df, 2024)
        assert total == pytest.approx(29000.0)
        assert days == 29

    def test_manglende_kolonne(self):
        df = pd.DataFrame({"Month": [1], "2023": [1000.0]})
        assert calculate_yearly_total_from_monthly_averages(df, 2024) == (0.0, 0, 0)

    def test_nan_ignoreres(self):
        df = pd.DataFrame({"Month": [1, 2], "2024": [1000.0, np.nan]})
        total, months, days = calculate_yearly_total_from_monthly_averages(df, 2024)
        assert total == pytest.approx(31000.0)
        assert months == 1


class TestExtractPointMonthlyMetrics:
    def test_kolonner_og_verdier(self):
        out = extract_point_monthly_metrics(two_point_year(), 2024)
        assert len(out) == 4
        rad = out[(out["point_id"] == "punkt_a") & (out["month"] == 1)].iloc[0]
        assert rad["avg_daily"] == pytest.approx(1000.0)
        assert rad["coverage_pct"] == pytest.approx(98.0)
        assert rad["ci_lower"] == pytest.approx(950.0)
        assert rad["ci_upper"] == pytest.approx(1050.0)
        assert rad["year"] == 2024
        assert rad["month_name"] == "Januar"

    def test_tom_input(self):
        assert extract_point_monthly_metrics({}, 2024).empty


class TestComputeMonthlyCoverageSummary:
    def test_med_data(self):
        out = compute_monthly_coverage_summary(two_point_year(), 2024, ["punkt_a", "punkt_b"])
        assert len(out) == 12
        jan = out[out["month"] == 1].iloc[0]
        assert jan["points_present"] == 2
        assert jan["points_present_pct"] == pytest.approx(100.0)
        assert jan["mean_coverage_pct"] == pytest.approx(96.5)
        assert jan["min_coverage_pct"] == pytest.approx(95.0)
        mars = out[out["month"] == 3].iloc[0]
        assert mars["points_present"] == 0
        assert pd.isna(mars["mean_coverage_pct"])

    def test_tom_input(self):
        out = compute_monthly_coverage_summary({}, 2024, ["punkt_a"])
        assert len(out) == 12
        assert (out["points_present"] == 0).all()
        assert (out["points_expected"] == 1).all()

    def test_avsluttet_aar_kan_vurderes_i_alle_maaneder(self):
        out = compute_monthly_coverage_summary(two_point_year(), 2024, ["punkt_a", "punkt_b"])
        assert out["is_assessable"].all()

    def test_fremtidige_maaneder_kan_ikke_vurderes(self):
        # Regresjon: måneder som ikke har inntruffet ble flagget som datahull
        neste_aar = date.today().year + 1
        out = compute_monthly_coverage_summary({}, neste_aar, ["punkt_a"])
        assert not out["is_assessable"].any()

    def test_innevaerende_aar_utelater_paagaaende_og_senere_maaneder(self):
        i_dag = date.today()
        out = compute_monthly_coverage_summary({}, i_dag.year, ["punkt_a"])
        vurderbare = out[out["is_assessable"]]["month"].tolist()
        assert vurderbare == list(range(1, i_dag.month))


class TestCoveragePivot:
    def test_pivot_form(self):
        metrics = extract_point_monthly_metrics(two_point_year(), 2024)
        pivot = coverage_pivot(metrics)
        assert pivot.loc["Januar", "punkt_a"] == pytest.approx(98.0)
        assert len(pivot.index) == 12  # reindeksert til alle månedsnavn

    def test_tom_input(self):
        assert coverage_pivot(pd.DataFrame()).empty


class TestTotalsWithUncertainty:
    def test_totaler_og_intervall(self):
        metrics = extract_point_monthly_metrics(two_point_year(), 2024)
        out = totals_with_uncertainty_from_metrics(metrics)
        jan = out[out["month"] == 1].iloc[0]
        assert jan["total"] == pytest.approx(1500.0 * 31)
        assert jan["total_lower"] == pytest.approx(1420.0 * 31)
        assert jan["total_upper"] == pytest.approx(1580.0 * 31)
        assert jan["coverage_pct"] == pytest.approx(96.5)

    def test_tom_input(self):
        assert totals_with_uncertainty_from_metrics(pd.DataFrame()).empty


class TestComputeMonthlyTotalsTable:
    def test_totaler_avrundes_til_heltall(self):
        df = pd.DataFrame({"Month": [1, 2], "2024": [1000.0, 1100.0]})
        out = compute_monthly_totals_table(df, [2024])
        assert out["2024"].dtype == "Int64"
        assert out["2024"].iloc[0] == 31000
        assert out["2024"].iloc[1] == 29 * 1100


class TestCalculateGrowthRates:
    def test_vekstkolonne(self):
        df = pd.DataFrame({"Month": [1], "2023": [100.0], "2024": [110.0]})
        out = calculate_growth_rates(df)
        assert out["Vekst 2023-2024 (%)"].iloc[0] == pytest.approx(10.0)

    def test_ett_aar_gir_ingen_vekstkolonner(self):
        df = pd.DataFrame({"Month": [1], "2024": [100.0]})
        out = calculate_growth_rates(df)
        assert list(out.columns) == ["Month", "2024"]


class TestCalculateSeasonalPatterns:
    def test_sesongsnitt(self):
        verdier = list(range(1, 13))  # jan=1 ... des=12
        df = pd.DataFrame({"Month": list(range(1, 13)), "2024": verdier})
        patterns = calculate_seasonal_patterns(df)
        assert patterns["2024"]["vinter_snitt"] == pytest.approx(np.mean([12, 1, 2]))
        assert patterns["2024"]["vår_snitt"] == pytest.approx(np.mean([3, 4, 5]))
        assert patterns["2024"]["sommer_snitt"] == pytest.approx(np.mean([6, 7, 8]))
        assert patterns["2024"]["høst_snitt"] == pytest.approx(np.mean([9, 10, 11]))

    def test_uten_month_kolonne(self):
        assert calculate_seasonal_patterns(pd.DataFrame({"x": [1]})) == {}
