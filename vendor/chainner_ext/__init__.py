# Vendored from chaiNNer-C (see vendor/PROVENANCE.md). Upstream adds nodes/impl to the DLL
# search path; here chainner_native.dll sits next to chainner_ext.pyd, and Windows resolves an
# extension module's DLL dependencies from the module's own directory.
from . import chainner_ext
from .chainner_ext import *

__doc__ = chainner_ext.__doc__
if hasattr(chainner_ext, "__all__"):
    __all__ = chainner_ext.__all__
