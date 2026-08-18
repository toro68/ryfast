"""Konstanter og konfigurasjon for Ryfast-trafikkdata (delt mellom app og CLI)."""

from datetime import date
from typing import Dict

URL = "https://trafikkdata-api.atlas.vegvesen.no"

TRAFFIC_POINTS = {
    "Ryfast (sum tunneler)": {
        "ids": [
            "99040V2725982",
            "00911V2725983",
            "10239V2725979",
            "62464V2725991",
            "92743V2726085",
            "25926V2725990",
        ],
        "description": "Sum av Ryfylketunnelen + Hundvågtunnelen (Ryfast totalt)",
        "opened": "2019-12-30 / 2020-04-22",
    },
    "Ryfylketunnelen": {
        "ids": ["99040V2725982", "00911V2725983"],
        "description": "Ryfylketunnelen - hovedforbindelse til Ryfylke",
        "opened": "2019-12-30",
    },
    "Hundvågtunnelen": {
        "ids": ["10239V2725979", "62464V2725991", "92743V2726085", "25926V2725990"],
        "description": "Hundvågtunnelen - forbindelse til Hundvåg og Eiganes",
        "opened": "2020-04-22",
    },
    "Bybrua": {
        "ids": {"Mot nord": ["17949V320695"], "Mot sør": ["54184V320694"]},
        "description": "Bybrua - historisk broforbindelse over Strømsteinsundet",
        "opened": "Historisk",
    },
}

HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE = ["10239V2725979", "92743V2726085"]
HUNDVAG_TUNNEL_RAMP_IDS = ["62464V2725991", "25926V2725990"]

MONTH_NAMES = [
    "Januar",
    "Februar",
    "Mars",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Desember",
]

# Første år med Ryfast-data; øvre grense følger kalenderen så listen ikke blir utdatert.
DATA_START_YEAR = 2019
YEAR_RANGE = range(DATA_START_YEAR, date.today().year + 2)
DEFAULT_YEARS = f"{date.today().year - 1},{date.today().year}"

API_MAX_RETRIES = 3
API_RETRY_DELAY = 1
API_CACHE_TTL = 24 * 3600
FULL_COVERAGE_TOL_PCT = 0.05  # tolerance to avoid float noise around 100%
ANOMALY_THRESHOLD_PCT = 20.0

MAX_BATCH_WORKERS = 6
MAX_WEEKLY_WORKERS = 12
API_ERROR_BUFFER_MAX = 200  # maks feiloppføringer i den trådsikre bufferen
API_ERROR_SESSION_MAX = 50  # maks feiloppføringer beholdt i session_state

COMPARE_YEARS = "Sammenlign år"
COMPARE_MONTHS = "Sammenlign måneder"
COMPARE_WEEKS = "Sammenlign uker"

QUERY_TEMPLATE = """
query {{
  trafficData(trafficRegistrationPointId: "{point_id}") {{
    volume {{
      average {{
        daily {{
          byMonth(year: {year}) {{
            month
            total {{
              volume {{
                average
                confidenceInterval {{
                  lowerBound
                  upperBound
                }}
              }}
              coverage {{
                percentage
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""

WEEKLY_QUERY_TEMPLATE = """
query {{
  trafficData(trafficRegistrationPointId: "{point_id}") {{
    volume {{
      byDay(from: "{from_date}", to: "{to_date}") {{
        edges {{
          node {{
            from
            to
            total {{
              volumeNumbers {{
                volume
              }}
              coverage {{
                percentage
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""

POINT_ID_LABELS: Dict[str, str] = {
    "99040V2725982": "Ryfylketunnelen (A)",
    "00911V2725983": "Ryfylketunnelen (B)",
    "10239V2725979": "Hundvågtunnelen (A)",
    "62464V2725991": "Hundvågtunnelen (pårampe?)",
    "92743V2726085": "Hundvågtunnelen (B)",
    "25926V2725990": "Hundvågtunnelen (pårampe?)",
    "17949V320695": "Bybrua (Mot nord)",
    "54184V320694": "Bybrua (Mot sør)",
}

# --- Sykkelregistrering, Nord-Jæren -------------------------------------------
# Punktene er hentet fra API-ets eget register (trafficRegistrationPoints med
# trafficRegistrationType=BICYCLE, fylke Rogaland) framfor å tastes inn manuelt:
# ID-er kan ikke gjettes fra vegreferanse, og en feil ID gir tomt svar uten feil.
# `operational=False` betyr at punktet er nedlagt (OUT_OF_SERVICE). Slike punkter
# har fortsatt historikk, men gir ingen tall for inneværende år.
BICYCLE_POINTS: Dict[str, Dict[str, object]] = {
    "21421B1828936": {
        "name": "Opstadvegen sykkel",
        "municipality": "Hå",
        "lat": 58.650771,
        "lon": 5.668631,
        "operational": True,
    },
    "28662B2604257": {
        "name": "Anda sykkel-begge retn.",
        "municipality": "Klepp",
        "lat": 58.775604,
        "lon": 5.642737,
        "operational": True,
    },
    "14779B1769080": {
        "name": "Kåsen sykkel- begge retn",
        "municipality": "Klepp",
        "lat": 58.746749,
        "lon": 5.632532,
        "operational": True,
    },
    "24783B2045151": {
        "name": "Orstad sykkel- begge retn.",
        "municipality": "Klepp",
        "lat": 58.798377,
        "lon": 5.72743,
        "operational": True,
    },
    "72727B282558": {
        "name": "Bråstein (sykkel)",
        "municipality": "Sandnes",
        "lat": 58.808369,
        "lon": 5.776337,
        "operational": True,
    },
    "29949B2713572": {
        "name": "Bærheim sykkel",
        "municipality": "Sandnes",
        "lat": 58.88326,
        "lon": 5.689183,
        "operational": True,
    },
    "46587B1727498": {
        "name": "Folkvord bru sykkel",
        "municipality": "Sandnes",
        "lat": 58.85098,
        "lon": 5.703123,
        "operational": True,
    },
    "59155B1685723": {
        "name": "Hogstad (sykkel)",
        "municipality": "Sandnes",
        "lat": 58.878372,
        "lon": 5.828406,
        "operational": True,
    },
    "89794B320138": {
        "name": "Hoveveien sykkel",
        "municipality": "Sandnes",
        "lat": 58.835729,
        "lon": 5.731553,
        "operational": True,
    },
    "51884B1868825": {
        "name": "Lura sykkel",
        "municipality": "Sandnes",
        "lat": 58.873749,
        "lon": 5.73371,
        "operational": False,
    },
    "43749B319868": {
        "name": "Soma skole (sykkel)",
        "municipality": "Sandnes",
        "lat": 58.86253,
        "lon": 5.701971,
        "operational": True,
    },
    "16851B2120991": {
        "name": "Somaveien sykkel",
        "municipality": "Sandnes",
        "lat": 58.872808,
        "lon": 5.724817,
        "operational": False,
    },
    "44930B2721303": {
        "name": "Flyplassvegen Sola Sykkel",
        "municipality": "Sola",
        "lat": 58.892621,
        "lon": 5.63084,
        "operational": True,
    },
    "13634B2721359": {
        "name": "Sømmevågen 2 Sykkel",
        "municipality": "Sola",
        "lat": 58.895981,
        "lon": 5.639604,
        "operational": True,
    },
    "08947B320223": {
        "name": "Bjergsted sykkel",
        "municipality": "Stavanger",
        "lat": 58.976726,
        "lon": 5.717356,
        "operational": False,
    },
    "01277B2094427": {
        "name": "Bybrua vest",
        "municipality": "Stavanger",
        "lat": 58.969186,
        "lon": 5.747731,
        "operational": True,
    },
    "01255B2094425": {
        "name": "Bybrua øst",
        "municipality": "Stavanger",
        "lat": 58.969309,
        "lon": 5.748117,
        "operational": True,
    },
    "75339B2422185": {
        "name": "Hillevåg sykkel",
        "municipality": "Stavanger",
        "lat": 58.938904,
        "lon": 5.744329,
        "operational": True,
    },
    "82609B1883476": {
        "name": "Kannik (sykkel)",
        "municipality": "Stavanger",
        "lat": 58.965675,
        "lon": 5.730462,
        "operational": True,
    },
    "75801B1859618": {
        "name": "Lassa sykkel",
        "municipality": "Stavanger",
        "lat": 58.959777,
        "lon": 5.699419,
        "operational": True,
    },
    "10887B320297": {
        "name": "Randabergveien sykkel",
        "municipality": "Stavanger",
        "lat": 58.975604,
        "lon": 5.710285,
        "operational": True,
    },
    "48028B1735081": {
        "name": "Revheimsvegen sykkel",
        "municipality": "Stavanger",
        "lat": 58.95164,
        "lon": 5.658076,
        "operational": True,
    },
    "73691B1835999": {
        "name": "Siddishallen Sykkel",
        "municipality": "Stavanger",
        "lat": 58.954643,
        "lon": 5.69501,
        "operational": True,
    },
    "04662B2863139": {
        "name": "Svartholen Sykkel",
        "municipality": "Stavanger",
        "lat": 58.912765,
        "lon": 5.684435,
        "operational": True,
    },
    "99274B3204701": {
        "name": "Sykkelstamvegen Forus nord",
        "municipality": "Stavanger",
        "lat": 58.890873,
        "lon": 5.714687,
        "operational": True,
    },
    "34304B3204701": {
        "name": "Sykkelstamvegen: Asser jåtten bru sør",
        "municipality": "Stavanger",
        "lat": 58.917125,
        "lon": 5.697605,
        "operational": True,
    },
    "35879B3204687": {
        "name": "Sykkelstamvegen:Asser jåtten bru sykkel vest",
        "municipality": "Stavanger",
        "lat": 58.917865,
        "lon": 5.698334,
        "operational": True,
    },
    "70394B2415370": {
        "name": "Vassbotnen sykkel",
        "municipality": "Stavanger",
        "lat": 58.890831,
        "lon": 5.713739,
        "operational": True,
    },
}

# Et sentralt og gjenkjennelig startpunkt på sykkelsiden. Standardvalget skal
# ikke avhenge av alfabetisk sortering av kommune og punktnavn.
BICYCLE_DEFAULT_POINT_ID = "99274B3204701"
BICYCLE_DEFAULT_OPENING_DATE = date(2025, 6, 16)

# Sykkeltellinger er små tall med kraftig ukesrytme, så døgnoppløsning er
# hovedvisningen. Terskelen brukes til å gråne ut dager med for lav dekning.
BICYCLE_MIN_COVERAGE_PCT = 50.0

# Første år med sykkeldata av brukbar kvalitet i registeret.
BICYCLE_DATA_START_YEAR = 2018

# byDay returnerer maks 100 døgn per side. Et helt år krever derfor
# paginering via pageInfo/after — uten den stopper grafen stille i april.
# Et år er 366 døgn = 4 sider; 6 gir margin uten å kunne løpe løpsk.
BICYCLE_MAX_PAGES = 6

# «Alle punkter» over flere år gir mange uavhengige kall (28 punkter × N år,
# hvert år opp mot 4 sider). Høyere tak enn bil-batchen fordi kallene er små
# og ventetiden dominerer.
MAX_BICYCLE_WORKERS = 12

BICYCLE_DAILY_QUERY_TEMPLATE = """
query {{
  trafficData(trafficRegistrationPointId: "{point_id}") {{
    volume {{
      byDay(from: "{from_date}", to: "{to_date}"{after_arg}) {{
        pageInfo {{
          hasNextPage
          endCursor
        }}
        edges {{
          node {{
            from
            total {{
              volumeNumbers {{
                volume
              }}
              coverage {{
                percentage
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""
