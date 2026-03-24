use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;
use unpii_core::{Engine, MaskMode, MaskOptions, PiiCategory, Span};

mod expressions;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// Python-visible Span object
#[pyclass(name = "Span")]
#[derive(Clone)]
struct PySpan {
    #[pyo3(get)]
    start: usize,
    #[pyo3(get)]
    end: usize,
    #[pyo3(get)]
    category: String,
}

#[pymethods]
impl PySpan {
    fn __repr__(&self) -> String {
        format!(
            "Span(start={}, end={}, category=\"{}\")",
            self.start, self.end, self.category
        )
    }
}

impl From<Span> for PySpan {
    fn from(s: Span) -> Self {
        PySpan {
            start: s.start,
            end: s.end,
            category: s.category.placeholder().trim_matches('<').trim_matches('>').to_string(),
        }
    }
}

fn parse_opts(mask: &str, mode: &str, ignore_groups: Option<Vec<String>>) -> MaskOptions {
    let mask_mode = MaskMode::from_str(mask);
    let paranoid = mode == "paranoid";
    let ignore = ignore_groups
        .unwrap_or_default()
        .iter()
        .map(|s| PiiCategory::from_label(s))
        .collect();

    MaskOptions {
        mode: mask_mode,
        paranoid,
        ignore_groups: ignore,
    }
}

#[pyfunction]
#[pyo3(signature = (text, *, mask="placeholder", mode="standard", ignore_groups=None))]
fn mask(
    text: &str,
    mask: &str,
    mode: &str,
    ignore_groups: Option<Vec<String>>,
) -> PyResult<String> {
    let engine = Engine::default_engine();
    let opts = parse_opts(mask, mode, ignore_groups);
    Ok(engine.mask(text, &opts))
}

#[pyfunction]
#[pyo3(signature = (text, *, mode="standard", ignore_groups=None))]
fn find_spans(
    text: &str,
    mode: &str,
    ignore_groups: Option<Vec<String>>,
) -> PyResult<Vec<PySpan>> {
    let engine = Engine::default_engine();
    let opts = parse_opts("placeholder", mode, ignore_groups);
    let spans = engine.find_spans(text, &opts);
    Ok(spans.into_iter().map(PySpan::from).collect())
}

#[pymodule]
fn unpii(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mask, m)?)?;
    m.add_function(wrap_pyfunction!(find_spans, m)?)?;
    m.add_class::<PySpan>()?;
    Ok(())
}
