# sa-ryfast

Små verktøy og en Streamlit-app for å hente og visualisere trafikkdata (Statens vegvesen GraphQL API) for Ryfast.

## Kom i gang

1. Opprett virtuelt miljø (anbefalt):
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Installer avhengigheter:
   - `python -m pip install -r requirements.txt`
3. Kjør appen:
   - `streamlit run traffic_data_app.py`

## Hjelpeskript

- `install_missing.sh`: Installerer/påser at avhengigheter finnes i et lokalt venv.
- `scripts/debug_api.py`: Enkel feilsøking av API-kall og responstider.
- `scripts/quick_fix.py`: Scratch/utkast for rask feilretting (ikke del av appen).
- `scripts/timeout_fix.py`: Scratch/utkast relatert til timeout-feilsøking (ikke del av appen).
- `compare_vegvesen_ferde.py`: Sammenlikner Vegvesen-tall mot Ferde-data (Excel i `docs/`).
