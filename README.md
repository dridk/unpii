# unpii

High-performance French medical text anonymization library. Rust core with Python bindings and native Polars integration.

Designed to process millions of documents efficiently.

## Installation

```bash
pip install unpii
```

## Quick Start

```python
import unpii

text = "Dr Martin au 06 12 34 56 78, email: martin@chu-brest.fr"

# Mask with placeholders (default)
unpii.mask(text)
# → "Dr <NOM> au <TELEPHONE>, email: <EMAIL>"

# Mask with stars
unpii.mask(text, mask="stars")
# → "Dr ***** au *****, email: *****"
```

## Detection Modes

Two detection levels: **standard** (reliable patterns) and **paranoid** (aggressive).

```python
# Standard: titles, known patterns, blacklisted names
unpii.mask("Dr Martin est ici")
# → "Dr <NOM> est ici"

unpii.mask("DUPONT Jean est ici")
# → "DUPONT Jean est ici"  (not detected in standard)

# Paranoid: also catches UPPERCASE Titlecase patterns, 5+ digit sequences, loose emails
unpii.mask("DUPONT Jean est ici", mode="paranoid")
# → "<NOM> est ici"
```

## Ignore Groups

Skip specific categories:

```python
unpii.mask("Dr Martin au 06 12 34 56 78", ignore_groups=["TELEPHONE"])
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

## Polars Integration

Native expression plugin — Polars handles parallelization automatically:

```python
import polars as pl
import unpii

df = pl.DataFrame({"text": [
    "Dr Martin au 06 12 34 56 78",
    "Email: joe@chu-brest.fr",
    "Maladie de Parkinson",
]})

df.with_columns(pl.col("text").unpii.mask())
# ┌─────────────────────────┐
# │ text                    │
# ╞═════════════════════════╡
# │ Dr <NOM> au <TELEPHONE> │
# │ Email: <EMAIL>          │
# │ Maladie de Parkinson    │  ← protected by whitelist
# └─────────────────────────┘

# With options
df.with_columns(
    pl.col("text").unpii.mask(mask="stars", mode="paranoid", ignore_groups=["TELEPHONE"])
)
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

## Whitelist

Medical eponyms (Parkinson, Alzheimer, Verneuil...) are protected from masking by a global whitelist, regardless of which group detected them.

## API Reference

```python
def mask(
    text: str,
    *,
    mask: str = "placeholder",       # "placeholder" or "stars"
    mode: str = "standard",          # "standard" or "paranoid"
    ignore_groups: list[str] | None = None,
) -> str: ...

def find_spans(
    text: str,
    *,
    mode: str = "standard",
    ignore_groups: list[str] | None = None,
) -> list[Span]: ...

# Span attributes: .start, .end, .category
```

## Performance

Rust core with compiled regex and Aho-Corasick automata. All rules and dictionaries are embedded in the binary — zero I/O at runtime.

Single-threaded: ~10ms per 7KB document (~100 docs/sec).
With Polars: automatic parallelization across all cores.

## License

MIT
