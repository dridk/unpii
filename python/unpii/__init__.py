from importlib.metadata import version

from .unpii import mask, find_spans, Span
from . import polars_ns as _  # trigger namespace registration on import

__version__ = version("unpii")
__all__ = ["mask", "find_spans", "Span", "__version__"]
