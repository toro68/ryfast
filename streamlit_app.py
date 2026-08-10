"""Entrypoint for Streamlit Cloud: `streamlit run streamlit_app.py`."""

# Må stå før alt som importerer pyarrow (streamlit gjør det): pyarrow 25 sin
# mimalloc-allokator segfaulter når Arrow allokerer første gang på Streamlits
# ScriptRunner-tråd. Se ryfast_app/arrow_compat.py for stakksporet.
import ryfast_app.arrow_compat  # noqa: F401  (setter allokator ved import)

from ryfast_app.app import main

if __name__ == "__main__":
    main()
