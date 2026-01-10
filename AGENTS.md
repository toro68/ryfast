# Repository Guidelines

## Project Structure & Module Organization

- `ryfast_app/app.py`: Streamlit app for fetching and visualizing traffic data from Statens vegvesen (GraphQL).
- `compare_vegvesen_ferde.py`: CLI tool for comparing Vegvesen counts with Ferde data (reads Excel in `docs/` and writes CSV).
- `scripts/debug_api.py`: Small script for troubleshooting API connectivity/response time.
- `docs/`: Input/output artifacts (e.g. `*.xlsx`, exported `*.csv`, markdown analyses).
- `requirements.txt`: Python dependencies.
- Virtualenvs are local-only: use `.venv/` and never commit it.

## Build, Test, and Development Commands

- Install deps: `python -m pip install -r requirements.txt`
- Run Streamlit app: `streamlit run ryfast_app/app.py`
- Compare datasets: `python compare_vegvesen_ferde.py --help`
- Debug API: `python scripts/debug_api.py`
- Quick syntax check (CI-style sanity): `python -m compileall -q .`

Notes:
- API calls hit `https://trafikkdata-api.atlas.vegvesen.no`; you need network access to run the app/scripts.

## Coding Style & Naming Conventions

- Language: Python 3.12+.
- Indentation: 4 spaces; prefer type hints for public helpers and CLI functions.
- Naming: `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants, `PascalCase` for dataclasses.
- Formatting/linting: no enforced tool in-repo; keep diffs small and consistent with existing style.

## Testing Guidelines

- No dedicated test suite currently.
- Before opening a PR, run `python -m compileall -q .` and (when relevant) manually smoke-test:
- `streamlit run ryfast_app/app.py` for UI changes
  - `python compare_vegvesen_ferde.py --years 2024,2025` for comparison changes

## Commit & Pull Request Guidelines

- Prefer short, imperative commit messages describing the change (e.g. “Clean up requirements.txt”, “Improve UI export”); avoid generic “update”.
- PRs should include: summary, how to run/verify, and any notable data/API assumptions.
- For Streamlit/UI changes, include screenshots or a brief screen recording.

## Agent-Specific Notes

- Do not commit generated data, caches, or virtualenv directories (`.venv/`, `__pycache__/`).
