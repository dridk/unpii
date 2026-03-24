from .unpii import mask, find_spans, Span
from . import polars_ns as _  # trigger namespace registration on import

__all__ = ["mask", "find_spans", "Span"]
