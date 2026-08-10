"""Omgåelse av en krasj i pyarrows mimalloc-allokator.

pyarrow 25.0.0 på macOS/arm64 segfaulter i `mi_thread_init` når Arrow
allokerer for første gang på en ny tråd. Streamlit kjører hver rerun i en egen
ScriptRunner-tråd, og alle diagrammer går gjennom Arrow, så krasjen treffer
midt i en helt vanlig opptegning:

    mi_thread_init + 956            <- SIGSEGV, KERN_INVALID_ADDRESS 0x18
    _mi_malloc_generic
    arrow::MimallocAllocator::AllocateAligned
    arrow::py::NumPyConverter::VisitNative<arrow::TimestampType>()
    streamlit ... altair_chart

Prosessen dør uten Python-traceback, så det ser ut som om appen bare henger
eller forsvinner. Å bytte til systemallokatoren fjerner krasjen. Kostnaden er
noe lavere allokeringsytelse i Arrow, som ikke er målbar for datamengdene her
(hundrevis til tusenvis av rader).

Timingen er avgjørende: valget må skje før pyarrow lastes. Derfor settes
miljøvariabelen ved import av denne modulen, og entrypointene importerer den
som sin aller første linje. Å kalle `pa.set_memory_pool()` fra `main()` er for
sent — da har mimalloc alt initialisert seg, og krasjen kommer likevel.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Settes ved import, før streamlit (og dermed pyarrow) er lastet.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")


def use_system_memory_pool() -> bool:
    """Sett Arrow til systemallokatoren. Returnerer True hvis den er aktiv.

    Feiler stille: krasjen gjelder én pyarrow-versjon på én plattform, og en
    framtidig versjon uten mimalloc skal ikke hindre appen fra å starte.
    """
    try:
        import pyarrow as pa

        if "mimalloc" not in pa.supported_memory_backends():
            return False
        pa.set_memory_pool(pa.system_memory_pool())
        return pa.default_memory_pool().backend_name == "system"
    except Exception as exc:
        logger.warning("Kunne ikke bytte Arrow-allokator: %s", exc)
        return False
