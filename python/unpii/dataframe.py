from __future__ import annotations

from .unpii import anonymize_batch


def anonymize_series(
    series,
    *,
    mask_from_columns: list | None = None,
    mask: list[str] | None = None,
    style: str = "placeholder",
    mode: str = "standard",
    ignore_groups: list[str] | None = None,
):
    """Anonymize PII in a Polars Series. Returns a new Series.

    Args:
        series: Polars Series of strings.
        mask_from_columns: List of Polars Series whose values are added as
            extra words to mask, per row.
        mask: Global list of words to mask on every row.
        style: "placeholder" or "stars".
        mode: "standard" or "paranoid".
        ignore_groups: PII categories to skip.
    """
    import polars as pl

    texts = series.to_list()

    col_lists = None
    if mask_from_columns:
        col_lists = [col.to_list() for col in mask_from_columns]

    results = anonymize_batch(
        texts,
        mask_from_columns=col_lists,
        mask=mask,
        style=style,
        mode=mode,
        ignore_groups=ignore_groups,
    )

    return pl.Series(series.name, results)


def anonymize_dataframe(
    df,
    column: str,
    *,
    mask_from_columns: list[str] | None = None,
    mask: list[str] | None = None,
    new_column: str | None = None,
    style: str = "placeholder",
    mode: str = "standard",
    ignore_groups: list[str] | None = None,
):
    """Anonymize PII in a DataFrame column. Returns a new DataFrame.

    Args:
        df: Polars DataFrame.
        column: Name of the text column to anonymize.
        mask_from_columns: Column names whose values are added as words
            to mask, per row.
        mask: Global list of words to mask on every row.
        new_column: Output column name. Defaults to overwriting the source column.
        style: "placeholder" or "stars".
        mode: "standard" or "paranoid".
        ignore_groups: PII categories to skip.
    """
    col_series = None
    if mask_from_columns:
        col_series = [df[col] for col in mask_from_columns]

    masked = anonymize_series(
        df[column],
        mask_from_columns=col_series,
        mask=mask,
        style=style,
        mode=mode,
        ignore_groups=ignore_groups,
    )

    target = new_column or column
    return df.with_columns(masked.alias(target))
