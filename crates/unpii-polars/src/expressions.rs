use std::sync::atomic::Ordering;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use serde::Deserialize;
use unpii_core::{Engine, MaskMode, MaskOptions, PiiCategory};

use crate::MAX_THREADS;

#[derive(Deserialize)]
struct MaskKwargs {
    #[serde(default = "default_mask")]
    mask: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default)]
    ignore_groups: Vec<String>,
    #[serde(default)]
    extra_count: usize,
}

fn default_mask() -> String {
    "placeholder".into()
}
fn default_mode() -> String {
    "standard".into()
}

#[polars_expr(output_type=String)]
fn mask_text(inputs: &[Series], kwargs: MaskKwargs) -> PolarsResult<Series> {
    let engine = Engine::default_engine();
    let base_opts = build_opts(&kwargs);

    let ca = inputs[0].str()?;

    // Collect extra columns as StringChunked
    let extra_cols: Vec<&StringChunked> = (1..=kwargs.extra_count)
        .map(|i| inputs[i].str())
        .collect::<PolarsResult<Vec<_>>>()?;

    let values: Vec<Option<&str>> = ca.into_iter().collect();

    let do_parallel = |values: &Vec<Option<&str>>| -> Vec<Option<String>> {
        if extra_cols.is_empty() {
            values
                .par_iter()
                .map(|opt_val| opt_val.map(|value| engine.mask(value, &base_opts)))
                .collect()
        } else {
            values
                .par_iter()
                .enumerate()
                .map(|(idx, opt_val)| {
                    opt_val.map(|value| {
                        let mut opts = MaskOptions {
                            mode: base_opts.mode.clone(),
                            paranoid: base_opts.paranoid,
                            ignore_groups: base_opts.ignore_groups.clone(),
                            extra: Vec::new(),
                        };
                        for col in &extra_cols {
                            if let Some(word) = col.get(idx) {
                                if !word.is_empty() {
                                    opts.extra.push(word.to_string());
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
    let results = if n > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        pool.install(|| do_parallel(&values))
    } else {
        do_parallel(&values)
    };

    let out: StringChunked = results.into_iter().collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}

fn build_opts(kwargs: &MaskKwargs) -> MaskOptions {
    let mode = MaskMode::from_str(&kwargs.mask);
    let paranoid = kwargs.mode == "paranoid";
    let ignore_groups = kwargs
        .ignore_groups
        .iter()
        .map(|s| PiiCategory::from_label(s))
        .collect();

    MaskOptions {
        mode,
        paranoid,
        ignore_groups,
        extra: Vec::new(),
    }
}
