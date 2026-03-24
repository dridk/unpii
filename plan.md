# Plan : Réécriture complète de `unpii`

## Contexte

Bibliothèque Python d'anonymisation de textes médicaux français, performante sur des millions de documents. Core en Rust, intégration native Polars (plugin expr), règles externalisées dans un fichier YAML pour faciliter la maintenance.

## Architecture

```
Cargo.toml                    # workspace
pyproject.toml                # maturin build
lang/
  fr/
    rules.yaml                # regex par groupe + modes standard/paranoid
    names.txt                 # blacklist : prénoms, noms à masquer
    cities.txt                # blacklist : noms de villes à masquer
    whitelist.txt             # whitelist : mots à protéger (noms de maladies...)
crates/
  unpii-core/                 # Rust pur, aucune dépendance Python/Polars
    src/
      lib.rs                  # re-exports
      category.rs             # enum PiiCategory + placeholder() + from_label()
      span.rs                 # Span {start, end, category} + resolve_overlaps()
      rules.rs                # charge YAML -> compile Regex par groupe
      keywords.rs             # charge txt -> Aho-Corasick par groupe
      engine.rs               # Engine (OnceLock) : 1 langue chargée, rules + keywords -> find_spans / mask
      masker.rs               # apply_mask (Stars / Placeholder)
  unpii-polars/               # cdylib : PyO3 + pyo3-polars
    src/
      lib.rs                  # #[pymodule] unpii { mask, find_spans }
      expressions.rs          # #[polars_expr] mask_text
python/
  unpii/
    __init__.py               # from .unpii import mask, find_spans
    _lib.py                   # LIB = Path(__file__).parent
    polars_ns.py              # @register_expr_namespace("unpii") -> register_plugin_function
tests/
  python/                     # on écrira les tests ensemble
```

## Pipeline de masquage

1. **Regex** (`rules.rs`) : pour chaque groupe du YAML, appliquer les regex compilées → `Vec<Span>`
2. **Blacklist** (`keywords.rs`) : Aho-Corasick sur les mots à masquer (prénoms, noms...) → `Vec<Span>` supplémentaires
3. **Whitelist** (`keywords.rs`) : Aho-Corasick sur les mots à protéger (noms de maladies...) → retirer les spans qui chevauchent un mot whitelisté
4. **Filtrer** les groupes dans `ignore_groups`
5. **Résoudre les chevauchements** : garder le span le plus long
6. **Masquer** : remplacer de la fin vers le début (`*****` ou `<CATEGORIE>`)

## Fichier de règles `lang/fr/rules.yaml`

```yaml
whitelist_files:
  - whitelist.txt              # global : protège contre tous les groupes

NOM:
  standard:
    patterns:
      - '(?i:Docteur|Dr\.?|Professeur|Prof\.?)\s+([A-ZÀ-Þ][a-zà-ÿ\-]+)'
    blacklist_files:
      - names.txt
  paranoid:
    patterns:
      - '\b[A-ZÀ-Þ]{2,}(?:[\-''][A-ZÀ-Þ]+)*\s+[A-ZÀ-Þ][a-zà-ÿ]{3,}\b'

TELEPHONE:
  standard:
    patterns:
      - '(?:(?:\+|00)33\s?[.\-]?|0)\(?[1-9]\)?(?:[\s.\-]?\d{2}){4}'

ADRESSE:
  standard:
    patterns:
      - '(?i)\d{1,4}\s*,?\s*(?:rue|avenue|boulevard|impasse|chemin|place)\s+[a-zA-ZÀ-ÿ\-'' ]{1,60}'
    blacklist_files:
      - cities.txt

EMAIL:
  standard:
    patterns:
      - '(?i)[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'

DATE:
  standard:
    patterns:
      - '\b(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:19|20)\d{2}\b'
      - '(?i)\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b'
```

**Modes** : `standard` et `paranoid`. Le mode paranoid active standard + paranoid (superset). Chaque groupe porte explicitement ses niveaux.

**Convention capture** : pas de groupe capturant → on masque le match entier. Avec groupes capturants → on masque uniquement les `()` capturés (pas les `(?:...)`). La regex porte l'info.

**Whitelist globale** : `whitelist_files` au top-level. Les mots whitelistés (Verneuil, Parkinson...) protègent contre le masquage quel que soit le groupe.

## Plugin Polars (architecture clé)

**Rust** (`expressions.rs`) :
```rust
#[polars_expr(output_type=String)]
fn mask_text(inputs: &[Series], kwargs: MaskKwargs) -> PolarsResult<Series> {
    let engine = Engine::default_engine();  // &'static, compilé une seule fois
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(|val, buf| {
        buf.push_str(&engine.mask(val, &opts));
    });
    Ok(out.into_series())
}
```

**Python** (`polars_ns.py`) :
```python
@pl.api.register_expr_namespace("unpii")
class UnpiiNamespace:
    def mask(self, mask="placeholder", ignore_groups=None) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB, function_name="mask_text",
            args=[self._expr], kwargs={...}, is_elementwise=True,
        )
```

Polars gère la parallélisation nativement (partitions par thread). Pas besoin de rayon.

## API Python cible

```python
import unpii
import polars as pl

# Langue par défaut : fr (embarquée dans le binaire via include_str!)
# Pour changer de langue (une seule à la fois, remplace le moteur) :
# unpii.init(lang="en")

# Niveau 1 : fonction simple
text = unpii.mask("Jean Dupond est ici")
text = unpii.mask("Jean Dupond est ici", mask="stars", mode="paranoid", ignore_groups=["PHONE"])

# Niveau 2 : inspection (dry-run)
spans = unpii.find_spans("Dr Martin au 06 12 34 56 78")
# -> [Span(start=0, end=9, category="NOM"), Span(start=13, end=27, category="TELEPHONE")]

# Niveau 3 : Polars natif (parallélisation automatique)
df.with_columns(pl.col("text").unpii.mask())
df.with_columns(pl.col("text").unpii.mask(mask="stars", ignore_groups=["PHONE"]))
```

### Signatures Python

```python
def mask(
    text: str,
    *,
    mask: str = "placeholder",       # "placeholder" (<NOM>) ou "stars" (*****)
    mode: str = "standard",          # "standard" ou "paranoid"
    ignore_groups: list[str] | None = None,
) -> str: ...

def find_spans(
    text: str,
    *,
    mode: str = "standard",
    ignore_groups: list[str] | None = None,
) -> list[Span]: ...

# Span est un objet avec attributs .start, .end, .category (pas un tuple)
```

## Dépendances Rust

**unpii-core** : `regex`, `aho-corasick`, `serde` + `serde_yaml`
**unpii-polars** : `unpii-core`, `pyo3`, `pyo3-polars`, `polars`, `serde`

## Ordre d'implémentation

1. Config workspace : `Cargo.toml`, `pyproject.toml`, crates `Cargo.toml`
2. `unpii-core` : category → span → masker → rules → keywords → engine → lib.rs
3. `lang/fr/rules.yaml` + `lang/fr/names.txt`, `cities.txt`, `whitelist.txt`
4. `unpii-polars` : lib.rs (pymodule) + expressions.rs (polars_expr)
5. `python/unpii/` : __init__.py, _lib.py, polars_ns.py
6. Build + smoke test
7. Tests unitaires (ensemble, avec plein de règles)
8. CLAUDE.md

## Vérification

```bash
cargo test -p unpii-core
uv sync && uv run maturin develop
uv run python -c "import unpii; print(unpii.mask('Dr Martin au 06 12 34 56 78'))"
uv run pytest tests/python/ -v
```
