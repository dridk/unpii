use crate::category::PiiCategory;
use crate::span::Span;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;

/// A single compiled pattern belonging to a category and mode level.
struct CompiledPattern {
    regex: Regex,
    category: PiiCategory,
    /// Number of capturing groups in this pattern.
    /// If > 0, only the captured groups are returned as spans.
    num_captures: usize,
}

/// The full compiled rule set, ready to scan text.
pub struct RuleSet {
    /// Patterns active in standard mode
    standard: Vec<CompiledPattern>,
    /// Additional patterns active only in paranoid mode
    paranoid: Vec<CompiledPattern>,
}

// --- YAML deserialization types ---

#[derive(Deserialize)]
struct RulesFile {
    #[serde(default)]
    whitelist_files: Vec<String>,
    #[serde(flatten)]
    groups: HashMap<String, GroupDef>,
}

#[derive(Deserialize)]
struct GroupDef {
    #[serde(default)]
    standard: Option<LevelDef>,
    #[serde(default)]
    paranoid: Option<LevelDef>,
}

#[derive(Deserialize)]
struct LevelDef {
    #[serde(default)]
    patterns: Vec<String>,
    #[serde(default)]
    blacklist_files: Vec<String>,
}

/// Parsed (but not yet compiled) rules from YAML.
pub struct ParsedRules {
    pub whitelist_files: Vec<String>,
    pub groups: Vec<ParsedGroup>,
}

pub struct ParsedGroup {
    pub category: PiiCategory,
    pub standard_patterns: Vec<String>,
    pub paranoid_patterns: Vec<String>,
    pub standard_blacklist_files: Vec<String>,
    pub paranoid_blacklist_files: Vec<String>,
}

impl ParsedRules {
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        let file: RulesFile =
            serde_yaml::from_str(yaml).map_err(|e| format!("YAML parse error: {}", e))?;

        let mut groups = Vec::new();
        for (name, def) in &file.groups {
            let category = PiiCategory::from_label(name);
            let mut pg = ParsedGroup {
                category,
                standard_patterns: Vec::new(),
                paranoid_patterns: Vec::new(),
                standard_blacklist_files: Vec::new(),
                paranoid_blacklist_files: Vec::new(),
            };
            if let Some(ref std) = def.standard {
                pg.standard_patterns = std.patterns.clone();
                pg.standard_blacklist_files = std.blacklist_files.clone();
            }
            if let Some(ref par) = def.paranoid {
                pg.paranoid_patterns = par.patterns.clone();
                pg.paranoid_blacklist_files = par.blacklist_files.clone();
            }
            groups.push(pg);
        }

        Ok(ParsedRules {
            whitelist_files: file.whitelist_files,
            groups,
        })
    }
}

impl RuleSet {
    /// Compile parsed rules into regex patterns.
    pub fn compile(parsed: &ParsedRules) -> Result<Self, String> {
        let mut standard = Vec::new();
        let mut paranoid = Vec::new();

        for group in &parsed.groups {
            for pat_str in &group.standard_patterns {
                let cp = compile_pattern(pat_str, group.category.clone())?;
                standard.push(cp);
            }
            for pat_str in &group.paranoid_patterns {
                let cp = compile_pattern(pat_str, group.category.clone())?;
                paranoid.push(cp);
            }
        }

        Ok(RuleSet { standard, paranoid })
    }

    /// Find all regex-based spans in text.
    /// If paranoid is true, both standard and paranoid patterns are used.
    pub fn find_spans(&self, text: &str, paranoid: bool) -> Vec<Span> {
        let mut spans = Vec::new();
        collect_spans(&self.standard, text, &mut spans);
        if paranoid {
            collect_spans(&self.paranoid, text, &mut spans);
        }
        spans
    }
}

fn compile_pattern(pat_str: &str, category: PiiCategory) -> Result<CompiledPattern, String> {
    let regex = Regex::new(pat_str).map_err(|e| format!("Invalid regex '{}': {}", pat_str, e))?;
    let num_captures = regex.captures_len() - 1; // captures_len includes group 0
    Ok(CompiledPattern {
        regex,
        category,
        num_captures,
    })
}

fn collect_spans(patterns: &[CompiledPattern], text: &str, spans: &mut Vec<Span>) {
    for cp in patterns {
        if cp.num_captures == 0 {
            // No capturing groups: mask entire match
            for mat in cp.regex.find_iter(text) {
                spans.push(Span::new(mat.start(), mat.end(), cp.category.clone()));
            }
        } else {
            // Has capturing groups: mask only the captured groups
            for caps in cp.regex.captures_iter(text) {
                for i in 1..=cp.num_captures {
                    if let Some(m) = caps.get(i) {
                        spans.push(Span::new(m.start(), m.end(), cp.category.clone()));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_yaml() {
        let yaml = r#"
whitelist_files:
  - whitelist.txt

NOM:
  standard:
    patterns:
      - '(?i:Dr\.?)\s+([A-Z][a-z]+)'
    blacklist_files:
      - names.txt
  paranoid:
    patterns:
      - '\b[A-Z]{2,}\s+[A-Z][a-z]{3,}\b'

EMAIL:
  standard:
    patterns:
      - '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
"#;
        let parsed = ParsedRules::from_yaml(yaml).unwrap();
        assert_eq!(parsed.whitelist_files, vec!["whitelist.txt"]);
        assert_eq!(parsed.groups.len(), 2);
    }

    #[test]
    fn test_capture_group_masking() {
        let yaml = r#"
NOM:
  standard:
    patterns:
      - '(?i:Dr\.?)\s+([A-Z][a-z]+)'
"#;
        let parsed = ParsedRules::from_yaml(yaml).unwrap();
        let ruleset = RuleSet::compile(&parsed).unwrap();
        let spans = ruleset.find_spans("Dr Martin est ici", false);
        assert_eq!(spans.len(), 1);
        // Should capture "Martin" only, not "Dr "
        assert_eq!(&"Dr Martin est ici"[spans[0].start..spans[0].end], "Martin");
    }

    #[test]
    fn test_full_match_no_capture() {
        let yaml = r#"
EMAIL:
  standard:
    patterns:
      - '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
"#;
        let parsed = ParsedRules::from_yaml(yaml).unwrap();
        let ruleset = RuleSet::compile(&parsed).unwrap();
        let spans = ruleset.find_spans("email: joe@test.fr fin", false);
        assert_eq!(spans.len(), 1);
        assert_eq!(&"email: joe@test.fr fin"[spans[0].start..spans[0].end], "joe@test.fr");
    }

    #[test]
    fn test_paranoid_mode() {
        let yaml = r#"
NOM:
  standard:
    patterns:
      - '(?i:Dr\.?)\s+([A-Z][a-z]+)'
  paranoid:
    patterns:
      - '\b[A-Z]{2,}\s+[A-Z][a-z]{3,}\b'
"#;
        let parsed = ParsedRules::from_yaml(yaml).unwrap();
        let ruleset = RuleSet::compile(&parsed).unwrap();

        // Standard: no match for "DUPONT Jean"
        let spans = ruleset.find_spans("DUPONT Jean", false);
        assert_eq!(spans.len(), 0);

        // Paranoid: matches
        let spans = ruleset.find_spans("DUPONT Jean", true);
        assert_eq!(spans.len(), 1);
    }
}
