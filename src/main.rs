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
use polars::prelude::*;
use std::time::Instant;
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
    fn load(model_id: &str) -> PiiResult<Self> {
        println!("Téléchargement du modèle {model_id}...");
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

        let device = Device::cuda_if_available(0).map_err(|e| PiiError::NlpEngine(e.to_string()))?;
        let device_name = if device.is_cuda() { "GPU (CUDA)" } else { "CPU" };
        println!("Device : {device_name}");

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

        println!("Modèle chargé !");
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

// ── Benchmark ───────────────────────────────────────────────────────────────

fn main() -> PiiResult<()> {
    let parquet_path = "medical_reports_1000_20260323_114953.parquet";

    // 1. Charger les CR médicaux depuis le fichier parquet
    println!("=== Chargement du parquet ===");
    let t0 = Instant::now();
    let df = LazyFrame::scan_parquet(parquet_path.into(), Default::default())
        .expect("impossible de lire le parquet")
        .collect()
        .expect("impossible de collecter le dataframe");
    println!("Parquet chargé en {:.2}s", t0.elapsed().as_secs_f64());
    println!("Shape : {:?}", df.shape());
    println!("Colonnes : {:?}\n", df.get_column_names());

    let report_col = df
        .column("report")
        .expect("colonne 'report' introuvable")
        .str()
        .expect("la colonne 'report' n'est pas de type string");

    let reports: Vec<&str> = report_col
        .into_iter()
        .filter_map(|v| v)
        .collect();

    let n_docs = reports.len();
    let total_chars: usize = reports.iter().map(|r| r.len()).sum();
    let avg_chars = total_chars / n_docs;
    println!("Documents : {n_docs}");
    println!("Taille totale : {total_chars} caractères ({:.1} Mo)", total_chars as f64 / 1_048_576.0);
    println!("Taille moyenne : {avg_chars} caractères/doc\n");

    // Afficher un extrait du premier document
    let sample = &reports[0][..reports[0].len().min(300)];
    println!("=== Extrait du 1er document ===");
    println!("{sample}...\n");

    // 2. Charger le modèle NER
    println!("=== Chargement du modèle NER ===");
    let ner_model = CamembertNerModel::load("almanach/camembertav2-base-ftb-ner")?;
    let base_engine = Box::new(SimpleNlpEngine::new(true));
    let ner_engine = CandleNerEngine::new(base_engine, Box::new(ner_model))?;

    let mut recognizers = default_recognizers();
    recognizers.push(Box::new(NerRecognizer::new("ner_camembert", vec![])));

    let policy = PolicyConfig::default();
    let analyzer = Analyzer::new(Box::new(ner_engine), recognizers, vec![], policy);
    let lang = Language::new("fr");
    let anon_config = AnonymizeConfig::default();

    // 3. Warmup (1er doc)
    println!("\n=== Warmup ===");
    let tw = Instant::now();
    let result = analyzer.analyze(reports[0], &lang)?;
    let _ = Anonymizer::anonymize(reports[0], &result.entities, &anon_config)?;
    println!("Warmup : {:.2}ms\n", tw.elapsed().as_secs_f64() * 1000.0);

    // 4. Benchmark complet sur tous les documents
    println!("=== Benchmark sur {n_docs} documents ===");
    let t_start = Instant::now();

    let mut total_entities = 0usize;
    let mut total_anon_chars = 0usize;
    let mut errors = 0usize;
    let mut timings_ms: Vec<f64> = Vec::with_capacity(n_docs);

    for (i, report) in reports.iter().enumerate() {
        let t_doc = Instant::now();
        match analyzer.analyze(report, &lang) {
            Ok(result) => {
                total_entities += result.entities.len();
                match Anonymizer::anonymize(report, &result.entities, &anon_config) {
                    Ok(anon) => total_anon_chars += anon.text.len(),
                    Err(_) => errors += 1,
                }
            }
            Err(_) => errors += 1,
        }
        let elapsed_ms = t_doc.elapsed().as_secs_f64() * 1000.0;
        timings_ms.push(elapsed_ms);

        if (i + 1) % 100 == 0 {
            let elapsed_total = t_start.elapsed().as_secs_f64();
            let docs_per_sec = (i + 1) as f64 / elapsed_total;
            println!(
                "  [{:>4}/{n_docs}] {:.1} docs/s | dernier: {:.1}ms",
                i + 1,
                docs_per_sec,
                elapsed_ms,
            );
        }
    }

    let total_secs = t_start.elapsed().as_secs_f64();

    // 5. Statistiques
    timings_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = timings_ms[n_docs / 2];
    let p95 = timings_ms[(n_docs as f64 * 0.95) as usize];
    let p99 = timings_ms[(n_docs as f64 * 0.99) as usize];
    let min_ms = timings_ms[0];
    let max_ms = timings_ms[n_docs - 1];
    let avg_ms = timings_ms.iter().sum::<f64>() / n_docs as f64;
    let docs_per_sec = n_docs as f64 / total_secs;

    println!("\n{}", "=".repeat(60));
    println!("=== RÉSULTATS DU BENCHMARK ===");
    println!("{}", "=".repeat(60));
    println!();
    println!("Documents traités   : {n_docs}");
    println!("Erreurs             : {errors}");
    println!("Entités détectées   : {total_entities}");
    println!("Entités/doc (moy.)  : {:.1}", total_entities as f64 / n_docs as f64);
    println!();
    println!("--- Temps par document ---");
    println!("  Moyenne : {avg_ms:>8.1}ms");
    println!("  Médiane : {p50:>8.1}ms");
    println!("  P95     : {p95:>8.1}ms");
    println!("  P99     : {p99:>8.1}ms");
    println!("  Min     : {min_ms:>8.1}ms");
    println!("  Max     : {max_ms:>8.1}ms");
    println!();
    println!("--- Débit ---");
    println!("  Temps total       : {total_secs:.1}s");
    println!("  Débit             : {docs_per_sec:.1} docs/s");
    println!("  Chars/s           : {:.0}", total_chars as f64 / total_secs);
    println!();

    // 6. Extrapolation à 1M documents
    let secs_for_1m = 1_000_000.0 / docs_per_sec;
    let hours_for_1m = secs_for_1m / 3600.0;
    let device_name = "RTX 3060 (GPU)";

    println!("=== EXTRAPOLATION À 1 000 000 DOCUMENTS ===");
    println!();
    println!("Sur cette machine ({device_name}, séquentiel) :");
    println!("  Temps estimé : {:.1}h ({:.0}s)", hours_for_1m, secs_for_1m);
    println!();

    // Estimation RTX 3060 (cette machine) — mesurée
    let rtx3060_docs_per_sec = docs_per_sec;

    // Estimation A40 :
    // - A40 : 299 TFLOPS FP16 (avec TF32), 37.4 TFLOPS FP32, 48 GB VRAM
    // - RTX 3060 : 12.7 TFLOPS FP32, 12 GB VRAM
    // - Ratio FP32 brut : ~3x
    // - Mais le goulot d'étranglement NER (batch=1) est souvent la latence mémoire
    //   et le overhead CPU (tokenization, post-processing), pas le compute pur.
    // - En pratique avec batch=1 : ~2x speedup réaliste par A40 vs RTX 3060
    // - Avec batching (batch=8-16 sur 48GB) : ~4-6x par A40 vs RTX 3060 batch=1
    // - 2x A40 : scaling ~1.8x (overhead inter-GPU)

    let a40_speedup_batch1 = 2.0;      // A40 vs RTX 3060, batch=1
    let a40_speedup_batched = 5.0;     // A40 vs RTX 3060, batch=8-16
    let dual_gpu_efficiency = 0.9;      // 2 GPU scaling efficiency

    let a40x1_batch1_dps = rtx3060_docs_per_sec * a40_speedup_batch1;
    let a40x2_batch1_dps = a40x1_batch1_dps * 2.0 * dual_gpu_efficiency;
    let a40x2_batched_dps = rtx3060_docs_per_sec * a40_speedup_batched * 2.0 * dual_gpu_efficiency;

    let a40x2_batch1_hours = 1_000_000.0 / a40x2_batch1_dps / 3600.0;
    let a40x2_batched_hours = 1_000_000.0 / a40x2_batched_dps / 3600.0;

    println!("Sur 2x NVIDIA A40 (48 GB chacune) :");
    println!();
    println!("  Scénario 1 — batch=1, séquentiel par GPU :");
    println!("    Speedup estimé  : {:.1}x vs RTX 3060", a40_speedup_batch1 * 2.0 * dual_gpu_efficiency);
    println!("    Débit           : {:.0} docs/s", a40x2_batch1_dps);
    println!("    Temps pour 1M   : {:.1}h", a40x2_batch1_hours);
    println!();
    println!("  Scénario 2 — batch=8-16, optimisé (recommandé) :");
    println!("    Speedup estimé  : {:.1}x vs RTX 3060", a40_speedup_batched * 2.0 * dual_gpu_efficiency);
    println!("    Débit           : {:.0} docs/s", a40x2_batched_dps);
    println!("    Temps pour 1M   : {:.1}h", a40x2_batched_hours);
    println!();
    println!("  Notes :");
    println!("    - Le batching est le levier principal (48 GB VRAM permet batch=16+)");
    println!("    - Le tokenizer CPU est souvent le goulot — utiliser rayon en parallèle");
    println!("    - Le modèle DeBERTa v2 base (110M params) tient facilement en VRAM");
    println!("    - Avec FP16 : ~2x speedup supplémentaire sur A40 (Tensor Cores)");

    Ok(())
}
