"""Tester for de rene dataframe-hjelperne i ui/charts.py."""

import pandas as pd
import pytest

from ryfast_app.ui.charts import (
    _long_year_df,
    _pairwise_period_comparison,
    _period_label_column,
    _year_columns,
)
from ryfast_app.ui.comparisons import _render_weekly_change_summary


@pytest.fixture
def month_df():
    return pd.DataFrame(
        {
            "Month": [1, 2],
            "Month Name": ["Januar", "Februar"],
            "2023": [100.0, 200.0],
            "2024": [110.0, None],
        }
    )


class TestYearColumns:
    def test_filtrerer_bort_metakolonner(self, month_df):
        assert _year_columns(month_df) == ["2023", "2024"]

    def test_ukedata(self):
        df = pd.DataFrame({"Week": ["Uke 1"], "2024": [1.0], "Volume": [2.0]})
        assert _year_columns(df) == ["2024"]


class TestLongYearDf:
    def test_smelter_og_dropper_nan(self, month_df):
        out = _long_year_df(month_df)
        assert set(out.columns) == {"Month", "Month Name", "År", "Trafikk"}
        assert len(out) == 3  # 2024/februar er NaN og droppes
        assert set(out["År"]) == {"2023", "2024"}


class TestPeriodLabelColumn:
    def test_maanedsnavn_foretrekkes(self, month_df):
        assert _period_label_column(month_df) == "Month Name"

    def test_uke(self):
        assert _period_label_column(pd.DataFrame({"Week": ["Uke 1"]})) == "Week"

    def test_ingen_periodekolonne(self):
        assert _period_label_column(pd.DataFrame({"x": [1]})) is None


class TestPairwisePeriodComparison:
    def test_endring_og_retning(self, month_df):
        out = _pairwise_period_comparison(month_df, "2023", "2024")
        assert len(out) == 1  # februar mangler 2024-verdi
        rad = out.iloc[0]
        assert rad["Baseline"] == pytest.approx(100.0)
        assert rad["Sammenligning"] == pytest.approx(110.0)
        assert rad["Endring"] == pytest.approx(10.0)
        assert rad["Endring (%)"] == pytest.approx(10.0)
        assert rad["Retning"] == "Opp"

    def test_manglende_kolonne_gir_tom_df(self, month_df):
        assert _pairwise_period_comparison(month_df, "2022", "2024").empty


class TestRenderWeeklyChangeSummary:
    def test_taaler_nullable_int64_med_na(self):
        # Regresjon: Int64-volum med pd.NA ga "boolean value of NA is ambiguous"
        df = pd.DataFrame(
            {
                "Week": ["Uke 1", "Uke 2", "Uke 3", "Uke 4"],
                "Volume": pd.array([1000, 1100, pd.NA, 1050], dtype="Int64"),
            }
        )
        _render_weekly_change_summary(df)  # skal ikke kaste

    def test_taaler_hull_i_indeksen(self):
        # Regresjon: idxmax/idxmin ble brukt med iloc; med hull i indeksen traff det feil rad
        df = pd.DataFrame(
            {
                "Week": ["Uke 1", "Uke 2", "Uke 3"],
                "Volume": [None, 900.0, 1200.0],
            }
        )
        _render_weekly_change_summary(df)  # skal ikke kaste
