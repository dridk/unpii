use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use unpii_core::{Engine, MaskMode, MaskOptions, PiiCategory};

#[derive(Deserialize)]
struct MaskKwargs {
    #[serde(default = "default_mask")]
    mask: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default)]
    ignore_groups: Vec<String>,
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
    let opts = build_opts(&kwargs);

    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(|value, buf| {
        let result = engine.mask(value, &opts);
        buf.push_str(&result);
    });
    Ok(out.into_series())
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
    }
}
