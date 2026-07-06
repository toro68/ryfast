# sa-ryfast

Små verktøy og en Streamlit-app for å hente og visualisere trafikkdata (Statens vegvesen GraphQL API) for Ryfast.

## Kom i gang

1. Opprett virtuelt miljø (anbefalt):
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Installer avhengigheter:
   - `python -m pip install -r requirements.txt`
   - For utvikling (tester og lint): `python -m pip install -r requirements-dev.txt`
3. Kjør appen:
   - `streamlit run streamlit_app.py`

## Utvikling

- Tester: `python -m pytest`
- Lint: `ruff check .`
- Kodestruktur og retningslinjer: se `AGENTS.md`

## Hjelpeskript

- `install_missing.sh`: Installerer/påser at avhengigheter finnes i et lokalt venv.
- `scripts/debug_api.py`: Enkel feilsøking av API-kall og responstider.
- `compare_vegvesen_ferde.py`: Sammenlikner Vegvesen-tall mot Ferde-data (Excel i `docs/`).
