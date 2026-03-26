use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum MaskMode {
    Stars,
    Placeholder,
}

impl MaskMode {
    pub fn from_str(s: &str) -> Self {
        match s {
            "stars" | "*" => MaskMode::Stars,
            _ => MaskMode::Placeholder,
        }
    }
}

impl Default for MaskMode {
    fn default() -> Self {
        MaskMode::Placeholder
    }
}

/// Apply masking to text given resolved (non-overlapping, sorted) spans.
/// Replaces from end to start to preserve byte offsets.
pub fn apply_mask(text: &str, spans: &[Span], mode: &MaskMode) -> String {
    let mut result = text.to_string();

    // Process spans from end to start
    for span in spans.iter().rev() {
        let replacement = match mode {
            MaskMode::Stars => "*****".to_string(),
            MaskMode::Placeholder => span.category.placeholder().to_string(),
        };
        result.replace_range(span.start..span.end, &replacement);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::PiiCategory;

    #[test]
    fn test_placeholder_mask() {
        let spans = vec![Span::new(0, 4, PiiCategory::Person)];
        let result = apply_mask("Jean Dupont", &spans, &MaskMode::Placeholder);
        assert_eq!(result, "<PERSON> Dupont");
    }

    #[test]
    fn test_stars_mask() {
        let spans = vec![Span::new(0, 4, PiiCategory::Person)];
        let result = apply_mask("Jean Dupont", &spans, &MaskMode::Stars);
        assert_eq!(result, "***** Dupont");
    }

    #[test]
    fn test_multiple_spans() {
        let spans = vec![
            Span::new(0, 4, PiiCategory::Person),
            Span::new(5, 11, PiiCategory::Person),
        ];
        let result = apply_mask("Jean Dupont est ici", &spans, &MaskMode::Placeholder);
        assert_eq!(result, "<PERSON> <PERSON> est ici");
    }
}
