"""Invarianter for konfigurasjon og målepunkt-definisjoner."""

from datetime import date

from ryfast_app.config import (
    DATA_START_YEAR,
    DEFAULT_YEARS,
    HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE,
    HUNDVAG_TUNNEL_RAMP_IDS,
    POINT_ID_LABELS,
    TRAFFIC_POINTS,
    YEAR_RANGE,
)


class TestYearRange:
    def test_starter_ved_datastart(self):
        assert YEAR_RANGE.start == DATA_START_YEAR == 2019

    def test_inkluderer_innevaerende_og_neste_aar(self):
        assert date.today().year in YEAR_RANGE
        assert date.today().year + 1 in YEAR_RANGE

    def test_default_years_er_fjoraar_og_i_aar(self):
        assert DEFAULT_YEARS == f"{date.today().year - 1},{date.today().year}"


class TestTrafficPointInvariants:
    def test_ryfast_sum_er_ryfylke_pluss_hundvaag(self):
        ryfast = set(TRAFFIC_POINTS["Ryfast (sum tunneler)"]["ids"])
        ryfylke = set(TRAFFIC_POINTS["Ryfylketunnelen"]["ids"])
        hundvaag = set(TRAFFIC_POINTS["Hundvågtunnelen"]["ids"])
        assert ryfast == ryfylke | hundvaag
        assert not ryfylke & hundvaag

    def test_rampe_pluss_uten_rampe_er_hele_hundvaag(self):
        hundvaag = set(TRAFFIC_POINTS["Hundvågtunnelen"]["ids"])
        uten = set(HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE)
        rampe = set(HUNDVAG_TUNNEL_RAMP_IDS)
        assert uten | rampe == hundvaag
        assert not uten & rampe

    def test_alle_tunnel_ider_har_etikett(self):
        for pid in TRAFFIC_POINTS["Ryfast (sum tunneler)"]["ids"]:
            assert pid in POINT_ID_LABELS

    def test_bybrua_ider_har_etikett(self):
        for ids in TRAFFIC_POINTS["Bybrua"]["ids"].values():
            for pid in ids:
                assert pid in POINT_ID_LABELS
