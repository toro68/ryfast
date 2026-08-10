"""Entrypoint brukt av Streamlit Cloud-oppsettet (main module = traffic_data_app.py).

Filnavnet er konfigurert i Streamlit Cloud-dashbordet, ikke i repoet, så det kan
ikke ses ved et søk gjennom kildefilene. Slett derfor ikke denne fila før
hovedmodulen er endret til `streamlit_app.py` i dashbordet — appen faller ned
med «Main module does not exist» i samme øyeblikk fila forsvinner.
"""

from ryfast_app.app import main

if __name__ == "__main__":
    main()
