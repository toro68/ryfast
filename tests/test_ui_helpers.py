"""Tester for de rene dataframe-hjelperne i ui/charts.py."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from ryfast_app.ui import banners as banners_mod
from ryfast_app.ui.banners import _assessable_only
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


def _coverage_row(month: int, month_name: str, present: int, mean_cov, assessable: bool) -> dict:
    return {
        "year": 2026,
        "month": month,
        "month_name": month_name,
        "points_present": present,
        "points_expected": 2,
        "mean_coverage_pct": mean_cov,
        "min_coverage_pct": mean_cov,
        "is_assessable": assessable,
    }


MONTH_NAMES_12 = [
    "Januar", "Februar", "Mars", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Desember",
]


class TestAssessableOnly:
    def test_dropper_ikke_vurderbare_perioder(self):
        df = pd.DataFrame(
            [
                _coverage_row(1, "Januar", 2, 100.0, True),
                _coverage_row(8, "August", 0, None, False),
            ]
        )
        out = _assessable_only(df)
        assert out["month"].tolist() == [1]

    def test_uten_kolonne_beholdes_alt(self):
        df = pd.DataFrame({"month": [1, 2], "points_present": [2, 2]})
        assert len(_assessable_only(df)) == 2


class TestRenderDataCoverageBanner:
    """Regresjon: banneret meldte datahull for måneder som ennå ikke er ferdige."""

    def _fange_meldinger(self, monkeypatch):
        meldinger = {"success": [], "warning": []}
        monkeypatch.setattr(banners_mod.st, "success", lambda m, **k: meldinger["success"].append(m))
        monkeypatch.setattr(banners_mod.st, "warning", lambda m, **k: meldinger["warning"].append(m))
        monkeypatch.setattr(banners_mod.st, "expander", lambda *a, **k: MagicMock())
        monkeypatch.setattr(banners_mod.st, "altair_chart", lambda *a, **k: None)
        monkeypatch.setattr(banners_mod.st, "dataframe", lambda *a, **k: None)
        return meldinger

    def test_ufullstendige_maaneder_gir_ikke_falsk_alarm(self, monkeypatch):
        meldinger = self._fange_meldinger(monkeypatch)
        # Perfekt dekning jan-jul 2026; aug-des har ikke inntruffet
        rows = [
            _coverage_row(m, navn, 2, 100.0, True) if m <= 7 else _coverage_row(m, navn, 0, None, False)
            for m, navn in enumerate(MONTH_NAMES_12, start=1)
        ]
        banners_mod.render_data_coverage_banner(pd.DataFrame(rows))
        assert not meldinger["warning"]
        assert len(meldinger["success"]) == 1
        assert "min 2/2" in meldinger["success"][0]

    def test_ekte_datahull_flagges_fortsatt(self, monkeypatch):
        meldinger = self._fange_meldinger(monkeypatch)
        rows = []
        for m, navn in enumerate(MONTH_NAMES_12, start=1):
            if m > 7:
                rows.append(_coverage_row(m, navn, 0, None, False))
            elif m == 3:
                rows.append(_coverage_row(m, navn, 1, 100.0, True))  # manglende punkt
            elif m == 5:
                rows.append(_coverage_row(m, navn, 2, 62.0, True))  # lav dekning
            else:
                rows.append(_coverage_row(m, navn, 2, 100.0, True))
        banners_mod.render_data_coverage_banner(pd.DataFrame(rows))
        assert not meldinger["success"]
        assert len(meldinger["warning"]) == 1
        tekst = meldinger["warning"][0]
        assert "Mars" in tekst and "Mai" in tekst
        # De ufullstendige månedene skal ikke nevnes
        assert "August" not in tekst and "Desember" not in tekst

    def test_ingen_vurderbare_perioder_gir_ingen_banner(self, monkeypatch):
        meldinger = self._fange_meldinger(monkeypatch)
        rows = [_coverage_row(m, navn, 0, None, False) for m, navn in enumerate(MONTH_NAMES_12, start=1)]
        banners_mod.render_data_coverage_banner(pd.DataFrame(rows))
        assert not meldinger["success"] and not meldinger["warning"]


class TestRenderPointBasisNote:
    def test_ufullstendige_maaneder_utloser_ikke_notat(self, monkeypatch):
        captions = []
        monkeypatch.setattr(banners_mod.st, "caption", lambda m, **k: captions.append(m))
        rows = [
            _coverage_row(m, navn, 2, 100.0, True) if m <= 7 else _coverage_row(m, navn, 0, None, False)
            for m, navn in enumerate(MONTH_NAMES_12, start=1)
        ]
        banners_mod.render_point_basis_note(pd.DataFrame(rows))
        assert captions == []

    def test_ekte_manglende_punkt_gir_notat(self, monkeypatch):
        captions = []
        monkeypatch.setattr(banners_mod.st, "caption", lambda m, **k: captions.append(m))
        rows = [
            _coverage_row(m, navn, 1 if m == 3 else 2, 100.0, True) if m <= 7 else _coverage_row(m, navn, 0, None, False)
            for m, navn in enumerate(MONTH_NAMES_12, start=1)
        ]
        banners_mod.render_point_basis_note(pd.DataFrame(rows))
        assert len(captions) == 1
        assert "1/2" in captions[0]


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
