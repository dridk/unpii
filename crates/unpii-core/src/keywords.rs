use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use crate::category::PiiCategory;
use crate::span::Span;

/// A compiled keyword set for blacklist or whitelist matching.
pub struct KeywordSet {
    ac: AhoCorasick,
    /// Parallel vec: entries[pattern_id] gives the category
    categories: Vec<PiiCategory>,
}

/// A compiled whitelist: just needs to know positions, no categories.
pub struct WhitelistSet {
    ac: AhoCorasick,
}

impl KeywordSet {
    /// Build from (word, category) pairs.
    /// Words are lowercased for case-insensitive matching.
    pub fn build(words: Vec<(String, PiiCategory)>) -> Option<Self> {
        if words.is_empty() {
            return None;
        }
        let patterns: Vec<String> = words.iter().map(|(w, _)| w.to_lowercase()).collect();
        let categories: Vec<PiiCategory> = words.into_iter().map(|(_, c)| c).collect();

        let ac = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("Failed to build Aho-Corasick automaton");

        Some(KeywordSet { ac, categories })
    }

    /// Find all keyword matches in text, checking word boundaries.
    pub fn find_spans(&self, text: &str) -> Vec<Span> {
        let mut spans = Vec::new();
        let text_lower = text.to_lowercase();
        let text_bytes = text_lower.as_bytes();

        for mat in self.ac.find_iter(&text_lower) {
            let start = mat.start();
            let end = mat.end();

            // Check word boundaries
            if !is_word_boundary(text_bytes, start, end) {
                continue;
            }

            let cat = self.categories[mat.pattern().as_usize()].clone();
            spans.push(Span::new(start, end, cat));
        }
        spans
    }
}

impl WhitelistSet {
    /// Build from a list of words to protect.
    pub fn build(words: Vec<String>) -> Option<Self> {
        if words.is_empty() {
            return None;
        }
        let patterns: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
        let ac = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("Failed to build whitelist Aho-Corasick automaton");

        Some(WhitelistSet { ac })
    }

    /// Return spans of whitelisted words found in text.
    pub fn find_protected_spans(&self, text: &str) -> Vec<(usize, usize)> {
        let text_lower = text.to_lowercase();
        let text_bytes = text_lower.as_bytes();
        let mut result = Vec::new();

        for mat in self.ac.find_iter(&text_lower) {
            if is_word_boundary(text_bytes, mat.start(), mat.end()) {
                result.push((mat.start(), mat.end()));
            }
        }
        result
    }
}

/// Check that a match is at word boundaries (not inside a larger word).
fn is_word_boundary(text: &[u8], start: usize, end: usize) -> bool {
    let before_ok = start == 0 || !is_word_char(text[start - 1]);
    let after_ok = end >= text.len() || !is_word_char(text[end]);
    before_ok && after_ok
}

fn is_word_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'\''
}

/// Parse a keyword file: one word per line, skip empty lines and comments (#).
pub fn parse_keyword_file(content: &str) -> Vec<String> {
    content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_matching() {
        let words = vec![
            ("Jean".to_string(), PiiCategory::Person),
            ("Marie".to_string(), PiiCategory::Person),
        ];
        let ks = KeywordSet::build(words).unwrap();
        let spans = ks.find_spans("Bonjour Jean et Marie");
        assert_eq!(spans.len(), 2);
    }

    #[test]
    fn test_word_boundary() {
        let words = vec![("Jean".to_string(), PiiCategory::Person)];
        let ks = KeywordSet::build(words).unwrap();
        // "Jeanne" should NOT match "Jean" due to word boundary
        let spans = ks.find_spans("Bonjour Jeanne");
        assert_eq!(spans.len(), 0);
    }

    #[test]
    fn test_whitelist() {
        let ws = WhitelistSet::build(vec!["Parkinson".to_string()]).unwrap();
        let protected = ws.find_protected_spans("Maladie de Parkinson");
        assert_eq!(protected.len(), 1);
        assert_eq!(protected[0], (11, 20));
    }

    #[test]
    fn test_parse_keyword_file() {
        let content = "Jean\n# comment\nMarie\n\nPierre\n";
        let words = parse_keyword_file(content);
        assert_eq!(words, vec!["Jean", "Marie", "Pierre"]);
    }

    #[test]
    fn test_case_insensitive() {
        let words = vec![("jean".to_string(), PiiCategory::Person)];
        let ks = KeywordSet::build(words).unwrap();
        let spans = ks.find_spans("Bonjour JEAN et jean");
        assert_eq!(spans.len(), 2);
    }
}
