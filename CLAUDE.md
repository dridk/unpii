# unpii - French Medical Text Anonymization

## Project Overview

High-performance Python library for anonymizing French medical text documents at scale (millions of docs).
Rust core (PyO3/maturin) with rayon parallelism. Rules externalized in YAML.

## Architecture

- `crates/unpii-core/` — Pure Rust library (no Python/Polars dependency)
- `crates/unpii-polars/` — cdylib: PyO3 bindings (`mask`, `find_spans`, `mask_batch`)
- `python/unpii/` — Python package (re-exports, `mask_series`/`mask_dataframe` wrappers)
- `lang/fr/` — French rules (rules.yaml, names.txt, cities.txt, whitelist.txt)
- `tests/python/` — pytest test suite

## Build & Test

```bash
cargo test -p unpii-core
uv sync
uv run maturin develop
uv run pytest tests/python/ -v
```

## Key Design Decisions

### Rules & Data
- Rules in YAML (`lang/fr/rules.yaml`), keyword lists in plain text files — all embedded in binary via `include_str!()`, zero I/O at runtime.
- Each group (NOM, TELEPHONE, EMAIL, DATE, ADRESSE, NIR, IBAN...) has `standard` and `paranoid` levels. Paranoid is a superset of standard.
- Blacklist files (names.txt, cities.txt) are per-group. Whitelist (whitelist.txt) is global.

### Capture Convention
- No capturing group `()` in regex → mask the entire match.
- With capturing groups → mask only the captured groups (not `(?:...)`).
- The regex itself carries the masking intent, no extra config needed.

### Engine
- `OnceLock<Engine>` singleton: YAML parsed, regex compiled, Aho-Corasick built once at first call, `&'static` thereafter.
- One language loaded at a time (default: fr).

### DataFrame Integration
- No Polars expression plugin. Instead: `anonymize_batch` (Rust/PyO3) processes `Vec<Option<String>>` with rayon parallelism.
- Python wrappers `anonymize_series` and `anonymize_dataframe` convert Polars Series to lists, call `anonymize_batch`, wrap results back.
- Polars is an optional dependency — `anonymize` and `anonymize_batch` work without it.

### Masking Pipeline
1. Regex detection (per group, per mode)
2. Blacklist (Aho-Corasick) → additional spans
3. Whitelist (Aho-Corasick) → remove spans overlapping whitelisted words
4. Filter by `ignore_groups`
5. Resolve overlapping spans (keep longest)
6. Mask from end to start (`*****` or `<CATEGORY>`)

## Python API

```python
import unpii

unpii.anonymize("Dr Martin au 06 12 34 56 78")
unpii.anonymize(text, style="stars", mode="paranoid", ignore_groups=["TELEPHONE"])
unpii.anonymize(text, mask=["Dupont"])  # custom words → <PII>
unpii.find_spans(text)  # dry-run, returns list[Span]

# DataFrame
df = unpii.anonymize_dataframe(df, "text")
df = unpii.anonymize_dataframe(df, "text", mask_from_columns=["nom", "ville"], mask=["Dupont"])
```

## Dependencies

**unpii-core**: `regex`, `aho-corasick`, `serde`, `serde_yaml`
**unpii-polars**: `unpii-core`, `pyo3`, `rayon`

## Rust Regex Crate Limitations
- No backreferences or variable-length lookbehinds.
- Use non-capturing groups `(?:...)` and structure regex so the match/captures are exactly what to mask.
