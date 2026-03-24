from __future__ import annotations

import polars as pl
from polars.plugins import register_plugin_function

from unpii._lib import LIB


@pl.api.register_expr_namespace("unpii")
class UnpiiNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def mask(
        self,
        *extra: pl.Expr,
        mask: str = "placeholder",
        mode: str = "standard",
        ignore_groups: list[str] | None = None,
    ) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            function_name="mask_text",
            args=[self._expr, *extra],
            kwargs={
                "mask": mask,
                "mode": mode,
                "ignore_groups": ignore_groups or [],
                "extra_count": len(extra),
            },
            is_elementwise=True,
        )
