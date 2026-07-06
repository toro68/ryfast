from __future__ import annotations

import os
import sys
from pathlib import Path


def run_streamlit_app() -> None:
    """
    Convenience entrypoint for environments that expect a root `main.py`.

    Deprecated: prefer `streamlit run streamlit_app.py`. Kept until it is
    confirmed that no deployment invokes `python main.py` directly.

    Equivalent to:
      streamlit run ryfast_app/app.py
    """

    from streamlit.web.cli import main as streamlit_main

    repo_root = Path(__file__).resolve().parent
    app_path = repo_root / "ryfast_app" / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Missing Streamlit app file: {app_path}")

    port = os.environ.get("PORT") or os.environ.get("STREAMLIT_SERVER_PORT") or "8501"
    address = os.environ.get("STREAMLIT_SERVER_ADDRESS") or "0.0.0.0"

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        f"--server.port={port}",
        f"--server.address={address}",
    ]
    streamlit_main()


if __name__ == "__main__":
    run_streamlit_app()

