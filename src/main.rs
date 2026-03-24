use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{
    Config as DebertaV2Config, DebertaV2NERModel, Id2Label,
};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use pii::{
    analyzer::Analyzer,
    anonymize::{AnonymizeConfig, Anonymizer},
    config::PolicyConfig,
    error::{PiiError, PiiResult},
    nlp::candle::{CandleNerEngine, CandleNerModel},
    nlp::SimpleNlpEngine,
    presets::default_recognizers,
    recognizers::ner::NerRecognizer,
    types::{EntityType, Language, NerSpan},
};
use rayon::prelude::*;
use regex::Regex;
use std::sync::{Arc, LazyLock};
use tokenizers::{PaddingParams, Tokenizer};

use std::io::Read;
use std::path::PathBuf;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "unpii", about = "Anonymise des fichiers texte (PII/NER)")]
struct Cli {
    /// Fichiers à anonymiser. Si absent, lit stdin.
    files: Vec<PathBuf>,

    /// Préfixe des fichiers de sortie (ex: "ano" → ano.fichier.txt)
    #[arg(long, default_value = "ano")]
    prefix: String,
}

// ── Modèle CamemBERT v2 NER ────────────────────────────────────────────────

struct CamembertNerModel {
    name: Arc<str>,
    model: DebertaV2NERModel,
    tokenizer: Tokenizer,
    id2label: Id2Label,
    device: Device,
}

impl CamembertNerModel {
    fn load(model_id: &str) -> PiiResult<Self> {
        eprintln!("Téléchargement du modèle {model_id}...");
        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());
        let api = Api::new().map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let api = api.repo(repo);

        let config_path = api.get("config.json").map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let tokenizer_path = api.get("tokenizer.json").map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let weights_path = api.get("model.safetensors").map_err(|e| PiiError::NlpEngine(e.to_string()))?;

        let config: DebertaV2Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path).map_err(|e| PiiError::NlpEngine(e.to_string()))?,
        )
        .map_err(|e| PiiError::NlpEngine(e.to_string()))?;

        let id2label = config
            .id2label
            .clone()
            .ok_or_else(|| PiiError::NlpEngine("id2label manquant".to_string()))?;

        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams::default()));

        let device = Device::Cpu;

        let weights_data = std::fs::read(&weights_path).map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let vb = VarBuilder::from_buffered_safetensors(
            weights_data,
            candle_transformers::models::debertav2::DTYPE,
            &device,
        )
        .map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let vb = vb.set_prefix("deberta");
        let model = DebertaV2NERModel::load(vb, &config, Some(id2label.clone()))
            .map_err(|e| PiiError::NlpEngine(e.to_string()))?;

        eprintln!("Modèle chargé !");
        Ok(Self { name: Arc::from(model_id), model, tokenizer, id2label, device })
    }
}

impl CandleNerModel for CamembertNerModel {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn infer(&self, _device: &Device, text: &str, _language: &Language) -> PiiResult<Vec<NerSpan>> {
        let device = &self.device;
        let encoding = self.tokenizer.encode(text, true).map_err(|e| PiiError::NlpEngine(e.to_string()))?;

        let input_ids = Tensor::stack(
            &[Tensor::new(encoding.get_ids(), device).map_err(|e| PiiError::NlpEngine(e.to_string()))?],
            0,
        )
        .map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let attention_mask = Tensor::stack(
            &[Tensor::new(encoding.get_attention_mask(), device).map_err(|e| PiiError::NlpEngine(e.to_string()))?],
            0,
        )
        .map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let token_type_ids = Tensor::stack(
            &[Tensor::new(encoding.get_type_ids(), device).map_err(|e| PiiError::NlpEngine(e.to_string()))?],
            0,
        )
        .map_err(|e| PiiError::NlpEngine(e.to_string()))?;

        let logits = self
            .model
            .forward(&input_ids, Some(token_type_ids), Some(attention_mask))
            .map_err(|e| PiiError::NlpEngine(e.to_string()))?;

        // Pas besoin de softmax : argmax(logits) == argmax(softmax(logits))
        // On utilise les logits bruts comme scores (l'ordre est préservé).
        let max_scores = logits.max(2).map_err(|e| PiiError::NlpEngine(e.to_string()))?.to_vec2::<f32>().map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let max_indices: Vec<Vec<u32>> = logits.argmax(2).map_err(|e| PiiError::NlpEngine(e.to_string()))?.to_vec2().map_err(|e| PiiError::NlpEngine(e.to_string()))?;

        let special_mask = encoding.get_special_tokens_mask();
        let offsets = encoding.get_offsets();
        let scores = &max_scores[0];
        let indices = &max_indices[0];

        let mut spans: Vec<NerSpan> = Vec::new();
        let mut current: Option<NerSpan> = None;

        for (idx, label_idx) in indices.iter().enumerate() {
            if special_mask[idx] == 1 {
                continue;
            }
            let label = match self.id2label.get(label_idx) {
                Some(l) => l.as_str(),
                None => continue,
            };
            if label == "O" {
                if let Some(span) = current.take() {
                    spans.push(span);
                }
                continue;
            }
            let (tag, entity_label) = split_label(label);
            let entity_type = match entity_label.and_then(label_to_entity) {
                Some(et) => et,
                None => {
                    if let Some(span) = current.take() {
                        spans.push(span);
                    }
                    continue;
                }
            };
            let (start, end) = offsets[idx];
            let score = scores[idx];
            let start_new = tag == Some("B");
            match current.as_mut() {
                Some(span) if !start_new && span.entity_type == entity_type && start <= span.end => {
                    span.end = end.max(span.end);
                    if score > span.score {
                        span.score = score;
                    }
                }
                _ => {
                    if let Some(span) = current.take() {
                        spans.push(span);
                    }
                    current = Some(NerSpan { entity_type, start, end, score, model: self.name.to_string() });
                }
            }
        }
        if let Some(span) = current.take() {
            spans.push(span);
        }
        Ok(spans)
    }
}

fn split_label(label: &str) -> (Option<&str>, Option<&str>) {
    if let Some(rest) = label.strip_prefix("B-") {
        return (Some("B"), Some(rest));
    }
    if let Some(rest) = label.strip_prefix("I-") {
        return (Some("I"), Some(rest));
    }
    (None, Some(label))
}

fn label_to_entity(label: &str) -> Option<EntityType> {
    match label {
        "Person" | "PER" => Some(EntityType::Person),
        "Organization" | "Company" | "ORG" => Some(EntityType::Organization),
        "Location" | "POI" | "LOC" => Some(EntityType::Location),
        _ => None,
    }
}

// ── Post-traitement emails ──────────────────────────────────────────────────

static PARTIAL_EMAIL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\S*<REDACTED>\S*@\S+|\S+@\S*<REDACTED>\S*").unwrap()
});

/// Corrige les emails partiellement anonymisés (ex: sacha@<REDACTED>.fr → <REDACTED>).
fn fix_partial_emails(text: &str) -> String {
    PARTIAL_EMAIL_RE.replace_all(text, "<REDACTED>").to_string()
}

// ── Anonymisation ───────────────────────────────────────────────────────────

fn anonymize_text(
    analyzer: &Analyzer,
    lang: &Language,
    anon_config: &AnonymizeConfig,
    text: &str,
) -> PiiResult<String> {
    let result = analyzer.analyze(text, lang)?;
    let anonymized = Anonymizer::anonymize(text, &result.entities, anon_config)?;
    Ok(fix_partial_emails(&anonymized.text))
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> PiiResult<()> {
    let cli = Cli::parse();

    let ner_model = CamembertNerModel::load("almanach/camembertav2-base-ftb-ner")?;
    let base_engine = Box::new(SimpleNlpEngine::new(true));
    let ner_engine = CandleNerEngine::new(base_engine, Box::new(ner_model))?;

    let mut recognizers = default_recognizers();
    recognizers.push(Box::new(NerRecognizer::new("ner_camembert", vec![])));

    let policy = PolicyConfig::default();
    let analyzer = Analyzer::new(Box::new(ner_engine), recognizers, vec![], policy);
    let lang = Language::new("fr");
    let anon_config = AnonymizeConfig::default();

    if cli.files.is_empty() {
        // Mode stdin → stdout
        let mut input = String::new();
        std::io::stdin().read_to_string(&mut input).map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let output = anonymize_text(&analyzer, &lang, &anon_config, &input)?;
        print!("{output}");
    } else {
        // Mode fichier(s) — traitement parallèle
        let total = cli.files.len();
        let t_global = std::time::Instant::now();

        // Lire tous les fichiers d'abord (I/O séquentiel)
        let inputs: Vec<(usize, &PathBuf, String)> = cli
            .files
            .iter()
            .enumerate()
            .map(|(i, path)| {
                let content = std::fs::read_to_string(path)
                    .map_err(|e| PiiError::NlpEngine(format!("{}: {e}", path.display())))?;
                Ok((i, path, content))
            })
            .collect::<PiiResult<Vec<_>>>()?;

        // Anonymiser en parallèle
        let results: Vec<PiiResult<(usize, &PathBuf, String, std::time::Duration)>> = inputs
            .par_iter()
            .map(|(i, path, content)| {
                let t = std::time::Instant::now();
                let output = anonymize_text(&analyzer, &lang, &anon_config, content)?;
                Ok((*i, *path, output, t.elapsed()))
            })
            .collect();

        // Écrire les résultats
        for result in results {
            let (i, path, output, elapsed) = result?;
            let filename = path.file_name().unwrap_or(path.as_os_str());
            let parent = path.parent().unwrap_or(std::path::Path::new("."));
            let out_path = parent.join(format!("{}.{}", cli.prefix, filename.to_string_lossy()));

            std::fs::write(&out_path, &output)
                .map_err(|e| PiiError::NlpEngine(format!("{}: {e}", out_path.display())))?;
            eprintln!("[{}/{}] {} → {} ({:.1}ms)", i + 1, total, path.display(), out_path.display(), elapsed.as_secs_f64() * 1000.0);
        }

        let total_elapsed = t_global.elapsed();
        eprintln!(
            "{total} fichier(s) anonymisé(s) en {:.1}s ({:.1}ms/doc)",
            total_elapsed.as_secs_f64(),
            total_elapsed.as_secs_f64() * 1000.0 / total as f64
        );
    }

    Ok(())
}
