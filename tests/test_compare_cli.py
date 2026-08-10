"""Tester for CLI-en: XLSX-lesing og parsing av Ferde-arket.

Arkene bygges i minnet, så testene er nettverks- og filfrie. Layouten speiler
`docs/2025_2024 trafikk og inntekt_Ryfast (1).xlsx`: år står bare på første
månedslinje, og en «Totalt»-rad avslutter hvert år.
"""

import zipfile
from pathlib import Path

import pytest

from compare_vegvesen_ferde import (
    parse_ferde_ryfast_sheet,
    read_xlsx_sheet_rows,
)

CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="xml" ContentType="application/xml"/>'
    "</Types>"
)


def _sheet_xml(rows: list[list[str]], shared: list[str]) -> str:
    """Bygg worksheet-XML: tall som <v>, tekst som indeks inn i sharedStrings."""
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for r_idx, row in enumerate(rows, start=1):
        out.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row):
            if value == "":
                continue
            col = chr(ord("A") + c_idx)
            if value.replace(".", "", 1).isdigit():
                out.append(f'<c r="{col}{r_idx}"><v>{value}</v></c>')
            else:
                out.append(f'<c r="{col}{r_idx}" t="s"><v>{shared.index(value)}</v></c>')
        out.append("</row>")
    out.append("</sheetData></worksheet>")
    return "".join(out)


def _shared_strings_xml(shared: list[str]) -> str:
    items = "".join(f"<si><t>{s}</t></si>" for s in shared)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"{items}</sst>"
    )


def _write_xlsx(path: Path, rows: list[list[str]], sheet_filename: str = "sheet1.xml", sheet_id: str = "1") -> Path:
    """Skriv en minimal .xlsx der arkfilnavnet kan avvike fra sheetId."""
    shared = sorted({c for row in rows for c in row if c and not c.replace(".", "", 1).isdigit()})
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("xl/sharedStrings.xml", _shared_strings_xml(shared))
        z.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
            ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets><sheet name="Sheet1" sheetId="{sheet_id}" r:id="rId1"/></sheets>'
            "</workbook>",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Target="worksheets/{sheet_filename}"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
            "</Relationships>",
        )
        z.writestr(f"xl/worksheets/{sheet_filename}", _sheet_xml(rows, shared))
    return path


FERDE_ROWS = [
    ["Ryfast - trafikk og inntekt", "", "", "", ""],
    ["År", "Måned", "Inntekt", "Passeringer", "Fritakspasseringer"],
    ["2025", "jan.", "24603757", "534667", "139483"],
    ["", "feb.", "22000000", "500000", "130000"],
    ["Totalt", "", "46603757", "1034667", "269483"],
    ["2026", "jan.", "25000000", "540000", "140000"],
    ["Totalt", "", "25000000", "540000", "140000"],
]


class TestReadXlsxSheetRows:
    def test_leser_celler_som_tekst(self, tmp_path):
        path = _write_xlsx(tmp_path / "enkel.xlsx", [["a", "1"], ["b", "2"]])
        assert read_xlsx_sheet_rows(path) == [["a", "1"], ["b", "2"]]

    def test_arkfilnavn_uavhengig_av_sheetid(self, tmp_path):
        # Regresjon: leseren antok xl/worksheets/sheet{sheetId}.xml. Ekte filer
        # kan ha sheetId=7 og filnavn sheet1.xml; da må r:id-relasjonen følges.
        path = _write_xlsx(
            tmp_path / "avvikende.xlsx",
            [["ok", "1"]],
            sheet_filename="sheet1.xml",
            sheet_id="7",
        )
        assert read_xlsx_sheet_rows(path) == [["ok", "1"]]

    def test_ukjent_sheet_index_gir_feil(self, tmp_path):
        path = _write_xlsx(tmp_path / "enkel.xlsx", [["a"]])
        with pytest.raises(ValueError, match="utenfor antall sheets"):
            read_xlsx_sheet_rows(path, sheet_index=3)


class TestParseFerdeRyfastSheet:
    def test_parser_maanedsrader_og_arver_aar(self):
        rows = parse_ferde_ryfast_sheet(FERDE_ROWS)
        assert len(rows) == 3
        jan = rows[0]
        assert (jan.year, jan.month) == (2025, 1)
        assert jan.income_nok == 24603757
        assert jan.passages_total == 534667
        assert jan.passages_exemptions == 139483
        # Året står bare på januar-linjen, men skal arves av februar
        assert (rows[1].year, rows[1].month) == (2025, 2)
        assert (rows[2].year, rows[2].month) == (2026, 1)

    def test_totalt_rad_tas_ikke_med(self):
        assert all(r.month in range(1, 13) for r in parse_ferde_ryfast_sheet(FERDE_ROWS))
        assert len(parse_ferde_ryfast_sheet(FERDE_ROWS)) == 3

    def test_manglende_header_gir_feil(self):
        with pytest.raises(ValueError, match="Fant ikke header-linje"):
            parse_ferde_ryfast_sheet([["År", "Noe annet"], ["2025", "jan."]])

    def test_ingen_maanedsrader_gir_feil(self):
        rows = [["År", "Måned", "Inntekt", "Passeringer", "Fritakspasseringer"]]
        with pytest.raises(ValueError, match="Fant ingen månedsrader"):
            parse_ferde_ryfast_sheet(rows)


class TestParseRealFerdeFile:
    def test_leser_arket_som_ligger_i_docs(self):
        path = Path("docs/2025_2024 trafikk og inntekt_Ryfast (1).xlsx")
        if not path.exists():
            pytest.skip("Ferde-arket ligger ikke i docs/")
        rows = parse_ferde_ryfast_sheet(read_xlsx_sheet_rows(path))
        assert rows
        assert all(1 <= r.month <= 12 for r in rows)
        assert all(r.passages_total >= 0 for r in rows)
