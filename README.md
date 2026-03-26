# unpii

High-performance French medical text anonymization library. Rust core with Python bindings.

Designed to process millions of documents efficiently. Inspired by [Micropot/incognito](https://github.com/Micropot/incognito)

## Installation

```bash
pip install unpii

# With Polars support
pip install unpii[polars]
```

## Quick Start

```python
import unpii

text = "Dr Martin au 06 12 34 56 78, email: martin@chu-brest.fr"

# Anonymize with placeholders (default)
unpii.anonymize(text)
# → "Dr <NOM> au <TELEPHONE>, email: <EMAIL>"

# Anonymize with stars
unpii.anonymize(text, style="stars")
# → "Dr ***** au *****, email: *****"
```

## Detection Modes

Two detection levels: **standard** (reliable patterns) and **paranoid** (aggressive).

```python
# Standard: titles, known patterns, blacklisted names
unpii.anonymize("Dr Martin est ici")
# → "Dr <NOM> est ici"

unpii.anonymize("DUPONT Jean est ici")
# → "DUPONT Jean est ici"  (not detected in standard)

# Paranoid: also catches UPPERCASE Titlecase patterns, 5+ digit sequences, loose emails
unpii.anonymize("DUPONT Jean est ici", mode="paranoid")
# → "<NOM> est ici"
```

## Custom Words to Mask

Pass additional words to mask per call. Useful when you know the patient's name:

```python
unpii.anonymize("bob dylan est ici", mask=["bob", "dylan"])
# → "<PII> <PII> est ici"
```

Case-insensitive with word boundary checks:

```python
unpii.anonymize("Bonjour Bob", mask=["bob"])
# → "Bonjour <PII>"
```

## Ignore Groups

Skip specific categories:

```python
unpii.anonymize("Dr Martin au 06 12 34 56 78", ignore_groups=["TELEPHONE"])
# → "Dr <NOM> au 06 12 34 56 78"
```

## Inspect Detected Spans

Dry-run mode to see what would be masked:

```python
for span in unpii.find_spans("Dr Martin au 06 12 34 56 78"):
    print(span)
# Span(start=3, end=9, category="NOM")
# Span(start=13, end=27, category="TELEPHONE")
```

## DataFrame Integration

`anonymize_dataframe` anonymizes a column in a Polars DataFrame:

```python
import polars as pl
import unpii

df = pl.DataFrame({"text": [
    "Dr Martin au 06 12 34 56 78",
    "Email: joe@chu-brest.fr",
    "Maladie de Parkinson",
]})

# Anonymize in place (overwrites the column)
df = unpii.anonymize_dataframe(df, "text")
# ┌─────────────────────────┐
# │ text                    │
# ╞═════════════════════════╡
# │ Dr <NOM> au <TELEPHONE> │
# │ Email: <EMAIL>          │
# │ Maladie de Parkinson    │  ← protected by whitelist
# └─────────────────────────┘

# Write to a new column
df = unpii.anonymize_dataframe(df, "text", new_column="text_anonymized")

# With options
df = unpii.anonymize_dataframe(df, "text", style="stars", mode="paranoid", ignore_groups=["TELEPHONE"])
```

### Per-row words to mask (`mask_from_columns`)

Pass column names whose values are added as words to mask, per row.
Useful when patient name/city are in structured columns:

```python
df = pl.DataFrame({
    "text": ["bob est ici", "alice va bien"],
    "nom": ["bob", "alice"],
})

df = unpii.anonymize_dataframe(df, "text", mask_from_columns=["nom"])
# ┌─────────────────┬───────┐
# │ text            ┆ nom   │
# ╞═════════════════╪═══════╡
# │ <PII> est ici   ┆ bob   │
# │ <PII> va bien   ┆ alice │
# └─────────────────┴───────┘
```

### Global words to mask (`mask`)

Words to mask on every row (e.g. the doctor who wrote all reports):

```python
df = unpii.anonymize_dataframe(df, "text", mask=["Dupont", "Cabinet Santé Plus"])
```

### Both combined

```python
df = unpii.anonymize_dataframe(df, "text",
    mask_from_columns=["nom", "ville"],
    mask=["Dupont"],
    style="stars",
)
```

### Low-level: `anonymize_series`

Operates on a Polars Series directly:

```python
masked = unpii.anonymize_series(
    df["text"],
    mask_from_columns=[df["nom"], df["ville"]],
    mask=["Dupont"],
)
df = df.with_columns(masked.alias("text_anonymized"))
```

### Batch processing: `anonymize_batch`

Operates on plain Python lists (no Polars dependency):

```python
results = unpii.anonymize_batch(["Dr Martin ici", "Email: a@b.fr"])
# → ["Dr <NOM> ici", "Email: <EMAIL>"]
```

## Threading

Control the number of threads used by `anonymize_batch`, `anonymize_series`, and `anonymize_dataframe`:

```python
unpii.set_max_threads(4)     # Use 4 threads
unpii.get_max_threads()      # → 4
unpii.set_max_threads(0)     # Use all available cores (default)
```

## Categories

| Group | Placeholder | Standard | Paranoid |
|-------|------------|----------|----------|
| NOM | `<NOM>` | Titles + name, blacklist | UPPERCASE/Titlecase patterns, initials |
| TELEPHONE | `<TELEPHONE>` | French phone numbers | — |
| EMAIL | `<EMAIL>` | Valid emails | Anything with `@` |
| DATE | `<DATE>` | DD/MM/YYYY, literal months, ISO | — |
| BIRTHDATE | `<BIRTHDATE>` | né(e) le + date | — |
| ADRESSE | `<ADRESSE>` | Street number + type + name | — |
| CODE_POSTAL | `<CODE_POSTAL>` | 5 digits + city name | — |
| NIR | `<NIR>` | French social security number | — |
| IBAN | `<IBAN>` | French IBAN | — |
| NUMBER | `<NUMBER>` | — | 5+ consecutive digits |
| PII | `<PII>` | Custom words passed via `mask=` | — |

## Whitelist

Medical eponyms (Parkinson, Alzheimer, Verneuil...) are protected from masking by a global whitelist, regardless of which group detected them.

## API Reference

```python
# Single text
def anonymize(text, *, style="placeholder", mode="standard", ignore_groups=None, mask=None) -> str
def find_spans(text, *, mode="standard", ignore_groups=None, mask=None) -> list[Span]

# Batch (plain Python lists, no Polars needed)
def anonymize_batch(texts, *, mask_from_columns=None, mask=None, style="placeholder", mode="standard", ignore_groups=None) -> list[str | None]

# DataFrame (requires polars)
def anonymize_dataframe(df, column, *, mask_from_columns=None, mask=None, new_column=None, style="placeholder", mode="standard", ignore_groups=None) -> DataFrame
def anonymize_series(series, *, mask_from_columns=None, mask=None, style="placeholder", mode="standard", ignore_groups=None) -> Series

# Threading
def set_max_threads(n: int) -> None   # 0 = all cores (default)
def get_max_threads() -> int

# Span attributes: .start, .end, .category
```

## Performance

Rust core with compiled regex and Aho-Corasick automata. All rules and dictionaries are embedded in the binary — zero I/O at runtime.

`anonymize_batch`, `anonymize_series`, and `anonymize_dataframe` use rayon for automatic parallelization across all cores.

## License

MIT

## See also

https://github.com/micropot/incognito
https://github.com/microsoft/presidio
