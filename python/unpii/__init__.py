from importlib.metadata import version

from .unpii import anonymize, find_spans, anonymize_batch, Span, set_max_threads, get_max_threads
from .dataframe import anonymize_series, anonymize_dataframe

__version__ = version("unpii")
__all__ = [
    "anonymize",
    "find_spans",
    "anonymize_batch",
    "anonymize_series",
    "anonymize_dataframe",
    "Span",
    "set_max_threads",
    "get_max_threads",
    "__version__",
]
