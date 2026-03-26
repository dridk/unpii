from importlib.metadata import version

from .unpii import mask, find_spans, Span, set_max_threads, get_max_threads
from . import polars_ns as _  # trigger namespace registration on import

__version__ = version("unpii")
__all__ = ["mask", "find_spans", "Span", "set_max_threads", "get_max_threads", "__version__"]
