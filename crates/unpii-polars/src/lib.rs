use std::sync::atomic::{AtomicUsize, Ordering};

use pyo3::prelude::*;
use rayon::prelude::*;
use unpii_core::{Engine, MaskMode, MaskOptions, PiiCategory, Span};

/// 0 means use all available cores (rayon default).
static MAX_THREADS: AtomicUsize = AtomicUsize::new(0);

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

fn parse_opts(style: &str, mode: &str, ignore_groups: Option<Vec<String>>, mask: Option<Vec<String>>) -> MaskOptions {
    let mask_mode = MaskMode::from_str(style);
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
        mask: mask.unwrap_or_default(),
    }
}

#[pyfunction]
#[pyo3(signature = (text, *, style="placeholder", mode="standard", ignore_groups=None, mask=None))]
fn anonymize(
    text: &str,
    style: &str,
    mode: &str,
    ignore_groups: Option<Vec<String>>,
    mask: Option<Vec<String>>,
) -> PyResult<String> {
    let engine = Engine::default_engine();
    let opts = parse_opts(style, mode, ignore_groups, mask);
    Ok(engine.mask(text, &opts))
}

#[pyfunction]
#[pyo3(signature = (text, *, mode="standard", ignore_groups=None, mask=None))]
fn find_spans(
    text: &str,
    mode: &str,
    ignore_groups: Option<Vec<String>>,
    mask: Option<Vec<String>>,
) -> PyResult<Vec<PySpan>> {
    let engine = Engine::default_engine();
    let opts = parse_opts("placeholder", mode, ignore_groups, mask);
    let spans = engine.find_spans(text, &opts);
    Ok(spans.into_iter().map(PySpan::from).collect())
}

#[pyfunction]
#[pyo3(signature = (texts, *, mask_from_columns=None, mask=None, style="placeholder", mode="standard", ignore_groups=None))]
fn anonymize_batch(
    py: Python<'_>,
    texts: Vec<Option<String>>,
    mask_from_columns: Option<Vec<Vec<Option<String>>>>,
    mask: Option<Vec<String>>,
    style: &str,
    mode: &str,
    ignore_groups: Option<Vec<String>>,
) -> PyResult<Vec<Option<String>>> {
    let engine = Engine::default_engine();
    let base_opts = parse_opts(style, mode, ignore_groups, mask);
    let mask_cols = mask_from_columns.unwrap_or_default();

    let results = py.allow_threads(|| {
        let do_parallel = || -> Vec<Option<String>> {
            if mask_cols.is_empty() {
                texts
                    .par_iter()
                    .map(|opt_val| {
                        opt_val.as_deref().map(|value| engine.mask(value, &base_opts))
                    })
                    .collect()
            } else {
                texts
                    .par_iter()
                    .enumerate()
                    .map(|(idx, opt_val)| {
                        opt_val.as_deref().map(|value| {
                            let mut opts = MaskOptions {
                                mode: base_opts.mode.clone(),
                                paranoid: base_opts.paranoid,
                                ignore_groups: base_opts.ignore_groups.clone(),
                                mask: base_opts.mask.clone(),
                            };
                            for col in &mask_cols {
                                if let Some(Some(word)) = col.get(idx) {
                                    if !word.is_empty() {
                                        opts.mask.push(word.clone());
                                    }
                                }
                            }
                            engine.mask(value, &opts)
                        })
                    })
                    .collect()
            }
        };

        let n = MAX_THREADS.load(Ordering::Relaxed);
        if n > 0 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("Failed to build rayon thread pool");
            pool.install(do_parallel)
        } else {
            do_parallel()
        }
    });

    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (n,))]
fn set_max_threads(n: usize) -> PyResult<()> {
    MAX_THREADS.store(n, Ordering::Relaxed);
    Ok(())
}

#[pyfunction]
fn get_max_threads() -> usize {
    MAX_THREADS.load(Ordering::Relaxed)
}

#[pymodule]
fn unpii(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(anonymize, m)?)?;
    m.add_function(wrap_pyfunction!(find_spans, m)?)?;
    m.add_function(wrap_pyfunction!(anonymize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(set_max_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_max_threads, m)?)?;
    m.add_class::<PySpan>()?;
    Ok(())
}
