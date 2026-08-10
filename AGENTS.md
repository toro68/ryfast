# Repository Guidelines

## Project Structure & Module Organization

- `ryfast_app/`: Streamlit-app for henting og visualisering av trafikkdata fra Statens vegvesen (GraphQL).
  - `app.py`: kun `init_session_state()` og `main()` (sidebar, orkestrering).
  - `config.py`: konstanter, målepunkt-ID-er, terskler, query-maler (delt med CLI).
  - `vegvesen_api.py`: streamlit-fri GraphQL-klient (`post_graphql`, retry, `iso_week_date_range`) — delt med CLI.
  - `api.py`: streamlit-laget (cache, trådsikker feilbuffer, parallell batch-/ukeshenting).
  - `processing.py`: radnivå-databehandling (summer, anomalier, punktmetrikker) — ren pandas, testbar.
  - `metrics.py`: dekningssammendrag, totaler, vekst, sesongmønstre — ren pandas, testbar.
  - `workflows.py`: `process_data_for_years/months/weeks` (driver `st.progress`).
  - `exports/`: `excel.py` og `pdf.py` (rapportbyggere).
  - `ui/`: `banners.py`, `charts.py`, `comparisons.py`, `export_section.py`, `tabs.py`.
- `streamlit_app.py`: eneste entrypoint for Streamlit Cloud (2-linjers shim mot `ryfast_app.app.main`).
- `compare_vegvesen_ferde.py`: CLI som sammenligner Vegvesen-tall med Ferde-data (leser Excel i `docs/`, skriver CSV). Bruker den delte klienten i `ryfast_app/vegvesen_api.py` og punkt-ID-ene i `ryfast_app/config.py`.
- `scripts/debug_api.py`: feilsøking av API-tilkobling/responstid.
- `tests/`: pytest-suite for ren databehandling, metrikker, API-klient og eksport.
- `docs/`: inn-/ut-artefakter (`*.xlsx`, eksporterte `*.csv`, markdown-analyser).
- `requirements.txt`: kjøreavhengigheter (Streamlit Cloud installerer denne). `requirements-dev.txt`: + pytest og ruff.
- Virtualenvs er lokale: bruk `.venv/` og ikke sjekk den inn.

## Build, Test, and Development Commands

- Installer avhengigheter: `python -m pip install -r requirements-dev.txt`
- Kjør Streamlit-appen: `streamlit run streamlit_app.py`
- Sammenlign datasett: `python compare_vegvesen_ferde.py --help`
- Debug API: `python scripts/debug_api.py`
- Tester: `python -m pytest`
- Lint: `ruff check .`
- Syntakssjekk: `python -m compileall -q .`

Notes:
- API-kall går mot `https://trafikkdata-api.atlas.vegvesen.no`; nettverkstilgang kreves for app/CLI (ikke for testene — de er nettverksfrie).

## Coding Style & Naming Conventions

- Språk: Python 3.12+.
- Innrykk: 4 mellomrom; foretrekk typehint for offentlige hjelpere og CLI-funksjoner.
- Navngiving: `snake_case` for funksjoner/variabler, `UPPER_SNAKE_CASE` for konstanter, `PascalCase` for dataklasser.
- Lint: `ruff check .` skal være grønn. Bred `except Exception` (BLE001) er kun tillatt i filene listet i `pyproject.toml` (eksport-/UI-vakter og session_state-flush), og skal alltid logge årsaken.
- Nye moduler skal ikke importere streamlit med mindre de er UI-/cache-lag (`api.py`, `ui/`, `exports/`, `workflows.py`, `app.py`). `config.py`, `vegvesen_api.py`, `processing.py` og `metrics.py` holdes streamlit-frie så CLI-en og testene kan bruke dem.

## Testing Guidelines

- Kjør `python -m pytest` og `ruff check .` før hver PR.
- Testene er nettverksfrie: API-atferd testes med monkeypatchet `requests.post`, eksport med monkeypatchet henting.
- Manuell røyktest i tillegg ved relevante endringer:
  - `streamlit run streamlit_app.py` for UI-endringer (alle tre moduser, alle fem faner, begge eksporter)
  - `python compare_vegvesen_ferde.py --years 2024,2025` for sammenligningsendringer

## Commit & Pull Request Guidelines

- Foretrekk korte, imperative commit-meldinger (f.eks. “Clean up requirements.txt”, “Improve UI export”); unngå generisk “update”.
- PR-er skal inneholde: oppsummering, hvordan kjøre/verifisere, og eventuelle data-/API-antakelser.
- For Streamlit/UI-endringer, legg ved skjermbilder eller kort skjermopptak.

## Agent-Specific Notes

- Ikke sjekk inn generert data, cacher eller virtualenv-kataloger (`.venv/`, `__pycache__/`).
- Flytting av `@st.cache_data`-funksjoner endrer cachenøkkelen (kald cache ved deploy) — ufarlig, men nevn det i commit-meldingen.
