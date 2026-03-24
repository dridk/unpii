use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{
    Config as DebertaV2Config, DebertaV2NERModel, Id2Label,
};
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
use tokenizers::{PaddingParams, Tokenizer};

// ── Modèle CamemBERT v2 NER ────────────────────────────────────────────────

struct CamembertNerModel {
    name: String,
    model: DebertaV2NERModel,
    tokenizer: Tokenizer,
    id2label: Id2Label,
    device: Device,
}

impl CamembertNerModel {
    fn load(model_id: &str, use_gpu: bool) -> PiiResult<Self> {
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

        let device = if use_gpu {
            Device::cuda_if_available(0).map_err(|e| PiiError::NlpEngine(e.to_string()))?
        } else {
            Device::Cpu
        };
        let device_name = if device.is_cuda() { "GPU (CUDA)" } else { "CPU" };
        eprintln!("Device : {device_name}");

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
        Ok(Self { name: model_id.to_string(), model, tokenizer, id2label, device })
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

        let probs = softmax(&logits, 2).map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let max_scores = probs.max(2).map_err(|e| PiiError::NlpEngine(e.to_string()))?.to_vec2::<f32>().map_err(|e| PiiError::NlpEngine(e.to_string()))?;
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
                    current = Some(NerSpan { entity_type, start, end, score, model: self.name.clone() });
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

// ── CLI ─────────────────────────────────────────────────────────────────────

fn main() -> PiiResult<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut use_gpu = true;
    let mut text_arg: Option<&str> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cpu" => use_gpu = false,
            "--gpu" => use_gpu = true,
            _ if text_arg.is_none() => text_arg = Some(&args[i]),
            _ => {}
        }
        i += 1;
    }

    let text = match text_arg {
        Some(t) => t,
        None => {
            eprintln!("Usage: {} [--cpu|--gpu] \"texte à anonymiser\"", args[0]);
            std::process::exit(1);
        }
    };

    let ner_model = CamembertNerModel::load("almanach/camembertav2-base-ftb-ner", use_gpu)?;
    let base_engine = Box::new(SimpleNlpEngine::new(true));
    let ner_engine = CandleNerEngine::new(base_engine, Box::new(ner_model))?;

    let mut recognizers = default_recognizers();
    recognizers.push(Box::new(NerRecognizer::new("ner_camembert", vec![])));

    let policy = PolicyConfig::default();
    let analyzer = Analyzer::new(Box::new(ner_engine), recognizers, vec![], policy);
    let lang = Language::new("fr");
    let anon_config = AnonymizeConfig::default();

    let result = analyzer.analyze(text, &lang)?;
    let anonymized = Anonymizer::anonymize(text, &result.entities, &anon_config)?;

    println!("{}", anonymized.text);

    Ok(())
}
