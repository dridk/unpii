use crate::category::PiiCategory;
use crate::keywords::{parse_keyword_file, KeywordSet, WhitelistSet};
use crate::masker::{apply_mask, MaskMode};
use crate::rules::{ParsedRules, RuleSet};
use crate::span::{resolve_overlaps, Span};
use std::collections::HashSet;
use std::sync::OnceLock;

// Embedded default language: French
const FR_RULES: &str = include_str!("../../../lang/fr/rules.yaml");
const FR_NAMES: &str = include_str!("../../../lang/fr/names.txt");
const FR_CITIES: &str = include_str!("../../../lang/fr/cities.txt");
const FR_REGIONS: &str = include_str!("../../../lang/fr/regions.txt");
const FR_WHITELIST: &str = include_str!("../../../lang/fr/whitelist.txt");

/// The compiled anonymization engine. Thread-safe, immutable after init.
pub struct Engine {
    rules: RuleSet,
    blacklists: Vec<KeywordSet>,
    whitelist: Option<WhitelistSet>,
}

static DEFAULT_ENGINE: OnceLock<Engine> = OnceLock::new();

/// Options for a mask/find_spans call.
pub struct MaskOptions {
    pub mode: MaskMode,
    pub paranoid: bool,
    pub ignore_groups: HashSet<PiiCategory>,
    pub mask: Vec<String>,
}

impl Default for MaskOptions {
    fn default() -> Self {
        MaskOptions {
            mode: MaskMode::Placeholder,
            paranoid: false,
            ignore_groups: HashSet::new(),
            mask: Vec::new(),
        }
    }
}

impl Engine {
    /// Get the default engine (French, embedded). Compiled once on first call.
    pub fn default_engine() -> &'static Engine {
        DEFAULT_ENGINE.get_or_init(|| {
            Self::build_fr().expect("Failed to build default French engine")
        })
    }

    fn build_fr() -> Result<Engine, String> {
        let parsed = ParsedRules::from_yaml(FR_RULES)?;
        let rules = RuleSet::compile(&parsed)?;

        // Build blacklists from the parsed group definitions
        let mut blacklists = Vec::new();
        let file_contents = resolve_files_fr();

        for group in &parsed.groups {
            let mut words = Vec::new();
            // Standard blacklist files
            for file_name in &group.standard_blacklist_files {
                if let Some(content) = file_contents.get(file_name.as_str()) {
                    for word in parse_keyword_file(content) {
                        words.push((word, group.category.clone()));
                    }
                }
            }
            // Paranoid blacklist files
            for file_name in &group.paranoid_blacklist_files {
                if let Some(content) = file_contents.get(file_name.as_str()) {
                    for word in parse_keyword_file(content) {
                        words.push((word, group.category.clone()));
                    }
                }
            }
            if let Some(ks) = KeywordSet::build(words) {
                blacklists.push(ks);
            }
        }

        // Build whitelist
        let mut whitelist_words = Vec::new();
        for file_name in &parsed.whitelist_files {
            if let Some(content) = file_contents.get(file_name.as_str()) {
                whitelist_words.extend(parse_keyword_file(content));
            }
        }
        let whitelist = WhitelistSet::build(whitelist_words);

        Ok(Engine {
            rules,
            blacklists,
            whitelist,
        })
    }

    /// Detect all PII spans in text.
    pub fn find_spans(&self, text: &str, opts: &MaskOptions) -> Vec<Span> {
        let mut spans = Vec::new();

        // 1. Regex detection
        spans.extend(self.rules.find_spans(text, opts.paranoid));

        // 2. Blacklist detection
        for ks in &self.blacklists {
            spans.extend(ks.find_spans(text));
        }

        // 3. Extra words (case-insensitive, word boundary)
        if !opts.mask.is_empty() {
            let text_lower = text.to_lowercase();
            for word in &opts.mask {
                let word_lower = word.to_lowercase();
                if word_lower.is_empty() {
                    continue;
                }
                let mut start = 0;
                while let Some(pos) = text_lower[start..].find(&word_lower) {
                    let abs_start = start + pos;
                    let abs_end = abs_start + word_lower.len();
                    // Check word boundaries
                    let before_ok = abs_start == 0
                        || !text.as_bytes()[abs_start - 1].is_ascii_alphanumeric();
                    let after_ok = abs_end >= text.len()
                        || !text.as_bytes()[abs_end].is_ascii_alphanumeric();
                    if before_ok && after_ok {
                        spans.push(Span::new(
                            abs_start,
                            abs_end,
                            PiiCategory::Custom("PII".to_string()),
                        ));
                    }
                    start = abs_end;
                }
            }
        }

        // 4. Whitelist: remove spans that overlap with protected words
        if let Some(ref ws) = self.whitelist {
            let protected = ws.find_protected_spans(text);
            spans.retain(|span| {
                !protected
                    .iter()
                    .any(|(ws, we)| span.start < *we && span.end > *ws)
            });
        }

        // 5. Filter by ignore_groups
        if !opts.ignore_groups.is_empty() {
            spans.retain(|s| !opts.ignore_groups.contains(&s.category));
        }

        // 6. Resolve overlaps
        resolve_overlaps(&mut spans)
    }

    /// Detect and mask PII in text.
    pub fn mask(&self, text: &str, opts: &MaskOptions) -> String {
        let spans = self.find_spans(text, opts);
        apply_mask(text, &spans, &opts.mode)
    }
}

/// Map embedded file names to their content for French.
fn resolve_files_fr() -> std::collections::HashMap<&'static str, &'static str> {
    let mut m = std::collections::HashMap::new();
    m.insert("names.txt", FR_NAMES);
    m.insert("cities.txt", FR_CITIES);
    m.insert("regions.txt", FR_REGIONS);
    m.insert("whitelist.txt", FR_WHITELIST);
    m
}
