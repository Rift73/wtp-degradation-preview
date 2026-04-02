import os
import sys
import glob
import importlib
import logging
import traceback

logger = logging.getLogger(__name__)

# Find all Python files in the current directory with names ending in '_degr.py'
modules = [
    os.path.basename(f)[:-3]
    for f in glob.glob(os.path.dirname(__file__) + "/*_degr.py")
]

# Import each module individually — one failing module shouldn't prevent
# all other degradations from loading.
# Store failures so the GUI can explain WHY a degradation is unavailable.
FAILED_MODULES: dict[str, str] = {}  # module_name -> traceback string

for module in sorted(modules):
    try:
        importlib.import_module(f".{module}", package=__name__)
    except Exception:
        tb = traceback.format_exc()
        FAILED_MODULES[module] = tb
        logger.error(
            "Failed to load degradation module '%s':\n%s\n"
            "This degradation will be unavailable. "
            "Check that all dependencies are installed.",
            module, tb,
        )

if FAILED_MODULES:
    names = ", ".join(FAILED_MODULES.keys())
    print(
        f"WARNING: {len(FAILED_MODULES)} degradation module(s) failed to load: {names}\n"
        f"See log for details. Other degradations are still available.",
        file=sys.stderr,
    )
