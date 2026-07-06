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
