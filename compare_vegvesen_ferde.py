"""Compare monthly Vegvesen counts with Ferde figures and export a CSV summary."""

import argparse
import calendar
import csv
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import xml.etree.ElementTree as ET

import requests


URL = "https://trafikkdata-api.atlas.vegvesen.no"
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1


TRAFFIC_POINT_GROUPS: Dict[str, List[str]] = {
    "Ryfast (sum tunneler, inkl pårampe)": [
        # Ryfylketunnelen (begge retninger)
        "99040V2725982",
        "00911V2725983",
        # Hundvågtunnelen (begge retninger + pårampe)
        "10239V2725979",
        "62464V2725991",
        "92743V2726085",
        "25926V2725990",
    ],
    "Hundvågtunnelen (inkl pårampe)": [
        "10239V2725979",
        "62464V2725991",
        "92743V2726085",
        "25926V2725990",
    ],
    "Hundvågtunnelen (uten pårampe)": ["10239V2725979", "92743V2726085"],
    "Ryfylketunnelen": ["99040V2725982", "00911V2725983"],
}


MONTH_NAME_TO_NUM = {
    "jan.": 1,
    "feb.": 2,
    "mar.": 3,
    "apr.": 4,
    "mai": 5,
    "jun.": 6,
    "jul.": 7,
    "aug.": 8,
    "sep.": 9,
    "okt.": 10,
    "nov.": 11,
    "des.": 12,
}


MONTH_QUERY = """
query {{
  trafficData(trafficRegistrationPointId: "{point_id}") {{
    volume {{
      average {{
        daily {{
          byMonth(year: {year}) {{
            month
            total {{
              volume {{ average }}
              coverage {{ percentage }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""


@dataclass(frozen=True)
class FerdeMonthRow:
    """One monthly row parsed from the Ferde spreadsheet."""

    year: int
    month: int
    income_nok: int
    passages_total: int
    passages_exemptions: int


def _gql(query: str, timeout_s: int = 30) -> Dict:
    last_error: Optional[Exception] = None
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.post(URL, json={"query": query}, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                raise RuntimeError(data["errors"][0].get("message", "GraphQL error"))
            return data["data"]
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt == API_MAX_RETRIES - 1:
                break
            time.sleep(API_RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"Vegvesen API-kall feilet etter {API_MAX_RETRIES} forsøk") from last_error


def fetch_vegvesen_monthly_totals(
    point_ids: List[str],
    year: int,
) -> Tuple[
    Dict[int, float],
    Dict[int, float],
    Dict[int, int],
    Dict[str, Dict[int, float]],
]:
    """
    Returnerer:
    - monthly_totals[month] = sum(ÅDT_måned * dager_i_måned) på tvers av punkter
        - monthly_coverage_avg[month] = snitt dekning (%) for måneden
            på tvers av punkter (der coverage finnes)
    - per_point_monthly[point_id][month] = (ÅDT_måned * dager_i_måned) for punktet
    """
    monthly_avg_sum: Dict[int, float] = {}
    monthly_cov: Dict[int, List[float]] = {}
    monthly_points_with_data: Dict[int, set] = {}
    per_point_monthly: Dict[str, Dict[int, float]] = {pid: {} for pid in point_ids}

    for pid in point_ids:
        query = MONTH_QUERY.format(point_id=pid, year=year)
        td = _gql(query).get("trafficData")
        if not td:
            continue
        rows = td["volume"]["average"]["daily"]["byMonth"] or []
        for row in rows:
            month = int(row["month"])
            avg_daily = row["total"]["volume"]["average"]
            if avg_daily is None:
                continue
            dim = calendar.monthrange(year, month)[1]
            month_total = float(avg_daily) * dim
            monthly_avg_sum[month] = monthly_avg_sum.get(month, 0.0) + month_total
            per_point_monthly[pid][month] = month_total
            monthly_points_with_data.setdefault(month, set()).add(pid)

            cov = row["total"]["coverage"]["percentage"]
            if cov is not None:
                monthly_cov.setdefault(month, []).append(float(cov))

    monthly_cov_avg = {m: (sum(v) / len(v)) for m, v in monthly_cov.items() if v}
    monthly_points_count = {m: len(s) for m, s in monthly_points_with_data.items()}
    return monthly_avg_sum, monthly_cov_avg, monthly_points_count, per_point_monthly


def _col_to_idx(col: str) -> int:
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _cell_to_rc(ref: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"^([A-Z]+)(\d+)$", ref)
    if not m:
        return None
    col, row = m.group(1), int(m.group(2))
    return row - 1, _col_to_idx(col)


def read_xlsx_sheet_rows(  # pylint: disable=too-many-locals,too-many-branches
    xlsx_path: Path, sheet_index: int = 0
) -> List[List[str]]:
    """
    Minimal XLSX-leser (uten openpyxl): returnerer en liste av rader (liste av celler som str).
    Støtter sharedStrings og numeriske verdier for enkle ark.
    """
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path) as z:
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = workbook.findall(".//main:sheets/main:sheet", ns)
        if not sheets:
            raise ValueError("Fant ingen sheets i workbook")
        if sheet_index >= len(sheets):
            raise ValueError(f"sheet_index={sheet_index} utenfor antall sheets={len(sheets)}")

        sheet_id = sheets[sheet_index].attrib["sheetId"]
        sheet_xml_path = f"xl/worksheets/sheet{sheet_id}.xml"

        shared: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss.findall("main:si", ns):
                text = "".join((t.text or "") for t in si.findall(".//main:t", ns))
                shared.append(text)

        sheet = ET.fromstring(z.read(sheet_xml_path))

        cells: Dict[Tuple[int, int], str] = {}
        max_r = 0
        max_c = 0
        for c in sheet.findall(".//main:c", ns):
            ref = c.attrib.get("r")
            if not ref:
                continue
            rc = _cell_to_rc(ref)
            if not rc:
                continue
            r, col = rc
            max_r = max(max_r, r)
            max_c = max(max_c, col)

            v_el = c.find("main:v", ns)
            if v_el is None:
                continue

            raw = v_el.text or ""
            if c.attrib.get("t") == "s":
                try:
                    val = shared[int(raw)]
                except Exception:
                    val = raw
            else:
                val = raw
            cells[(r, col)] = val

        rows: List[List[str]] = []
        for r in range(0, max_r + 1):
            row: List[str] = []
            for col in range(0, max_c + 1):
                row.append(cells.get((r, col), "").strip())
            rows.append(row)
        return rows


def parse_ferde_ryfast_sheet(rows: List[List[str]]) -> List[FerdeMonthRow]:
    """
    Leser tabellen som ligger i `docs/2025_2024 trafikk og inntekt_Ryfast (1).xlsx`.
    Forventer layout med:
      A: år (kun på første månedslinje), B: måned, C: inntekt, D: passeringer, E: fritakspasseringer
    """
    result: List[FerdeMonthRow] = []

    # Finn header-linje
    header_idx = None
    for i, row in enumerate(rows):
        row_join = " ".join(x for x in row if x)
        if "Måned" in row_join and "Passeringer" in row_join and "Fritakspasseringer" in row_join:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Fant ikke header-linje med Måned/Passeringer/Fritakspasseringer")

    current_year: Optional[int] = None
    for row in rows[header_idx + 1 :]:
        # Tom rad
        if not any(cell for cell in row):
            continue

        a = row[0] if len(row) > 0 else ""
        b = row[1] if len(row) > 1 else ""
        c = row[2] if len(row) > 2 else ""
        d = row[3] if len(row) > 3 else ""
        e = row[4] if len(row) > 4 else ""

        if a.isdigit():
            current_year = int(a)

        if a.lower() == "totalt":
            current_year = None
            continue

        if current_year is None:
            continue

        month_key = b.strip().lower()
        if month_key not in MONTH_NAME_TO_NUM:
            continue

        def as_int(x: str) -> int:
            x = x.strip()
            if not x:
                return 0
            try:
                return int(float(x))
            except (ValueError, TypeError):
                return 0

        income = as_int(c)
        passages = as_int(d)
        exemptions = as_int(e)

        result.append(
            FerdeMonthRow(
                year=current_year,
                month=MONTH_NAME_TO_NUM[month_key],
                income_nok=income,
                passages_total=passages,
                passages_exemptions=exemptions,
            )
        )

    if not result:
        raise ValueError("Fant ingen månedsrader for Ferde i arket")
    return result


def write_csv(path: Path, header: List[str], rows: Iterable[Dict[str, object]]) -> None:
    """Write semicolon-separated output rows to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter=";")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in header})


def main() -> int:
    """Run the comparison CLI and export monthly comparison rows."""

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ferde-xlsx",
        default="docs/2025_2024 trafikk og inntekt_Ryfast (1).xlsx",
        help="Path til Ferde-Excel",
    )
    ap.add_argument(
        "--group",
        default="Ryfast (sum tunneler, inkl pårampe)",
        choices=sorted(TRAFFIC_POINT_GROUPS.keys()),
        help="Hvilken gruppe av målepunkter som skal summeres fra Vegvesen",
    )
    ap.add_argument("--years", default="2024,2025", help="Komma-separert liste med år")
    ap.add_argument(
        "--out",
        default="docs/vegvesen_vs_ferde_2024_2025.csv",
        help="Output CSV (semicolon-separert)",
    )
    args = ap.parse_args()

    ferde_path = Path(args.ferde_xlsx)
    out_path = Path(args.out)

    try:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    except ValueError as exc:
        raise SystemExit("Ugyldig --years. Bruk f.eks. 2024,2025") from exc
    if not years:
        raise SystemExit("Ugyldig --years. Oppgi minst ett år.")

    ferde_rows = parse_ferde_ryfast_sheet(read_xlsx_sheet_rows(ferde_path))
    ferde_by_ym: Dict[Tuple[int, int], FerdeMonthRow] = {
        (r.year, r.month): r for r in ferde_rows
    }

    point_ids = TRAFFIC_POINT_GROUPS[args.group]

    out_rows: List[Dict[str, object]] = []
    for year in years:  # pylint: disable=too-many-locals
        veg_monthly, veg_cov, veg_points_count, _ = fetch_vegvesen_monthly_totals(point_ids, year)
        for month in range(1, 13):
            ferde = ferde_by_ym.get((year, month))
            veg_total = veg_monthly.get(month)

            ferde_total = ferde.passages_total if ferde else None
            ferde_ex = ferde.passages_exemptions if ferde else None
            ferde_pay = (
                (ferde_total - ferde_ex)
                if (ferde_total is not None and ferde_ex is not None)
                else None
            )
            ferde_income = ferde.income_nok if ferde else None

            diff = (
                (veg_total - ferde_total)
                if (veg_total is not None and ferde_total is not None)
                else None
            )
            ratio = (veg_total / ferde_total) if (veg_total is not None and ferde_total) else None

            out_rows.append(
                {
                    "område": args.group,
                    "år": year,
                    "måned": month,
                    "vegvesen_tellinger": int(round(veg_total)) if veg_total is not None else "",
                    "vegvesen_dekning_snitt_prosent": (
                        round(veg_cov.get(month), 1) if month in veg_cov else ""
                    ),
                    "vegvesen_antall_punkt_med_data": veg_points_count.get(month, 0),
                    "vegvesen_antall_punkt_total": len(point_ids),
                    "ferde_passeringer": ferde_total if ferde_total is not None else "",
                    "ferde_fritakspasseringer": ferde_ex if ferde_ex is not None else "",
                    "ferde_betalende": ferde_pay if ferde_pay is not None else "",
                    "ferde_inntekt_nok": ferde_income if ferde_income is not None else "",
                    "diff_vegvesen_minus_ferde": int(round(diff)) if diff is not None else "",
                    "ratio_vegvesen_div_ferde": round(ratio, 4) if ratio is not None else "",
                }
            )

    header = [
        "område",
        "år",
        "måned",
        "vegvesen_tellinger",
        "vegvesen_dekning_snitt_prosent",
        "vegvesen_antall_punkt_med_data",
        "vegvesen_antall_punkt_total",
        "ferde_passeringer",
        "ferde_fritakspasseringer",
        "ferde_betalende",
        "ferde_inntekt_nok",
        "diff_vegvesen_minus_ferde",
        "ratio_vegvesen_div_ferde",
    ]
    write_csv(out_path, header, out_rows)
    print(f"Skrev {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
