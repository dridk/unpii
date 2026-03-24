use crate::category::PiiCategory;

#[derive(Debug, Clone)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub category: PiiCategory,
}

impl Span {
    pub fn new(start: usize, end: usize, category: PiiCategory) -> Self {
        Self { start, end, category }
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }
}

/// Remove overlapping spans, keeping the longest when two overlap.
/// Input is sorted by start position, then by length descending.
pub fn resolve_overlaps(spans: &mut Vec<Span>) -> Vec<Span> {
    if spans.is_empty() {
        return Vec::new();
    }

    spans.sort_by(|a, b| a.start.cmp(&b.start).then(b.len().cmp(&a.len())));

    let mut result: Vec<Span> = Vec::new();
    for span in spans.drain(..) {
        if let Some(last) = result.last() {
            if span.start < last.end {
                // Overlap: keep the one already in result (it's longer or equal due to sort)
                continue;
            }
        }
        result.push(span);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_overlap() {
        let mut spans = vec![
            Span::new(0, 5, PiiCategory::Nom),
            Span::new(10, 15, PiiCategory::Email),
        ];
        let resolved = resolve_overlaps(&mut spans);
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn test_overlap_keeps_longest() {
        let mut spans = vec![
            Span::new(0, 10, PiiCategory::Nom),
            Span::new(5, 8, PiiCategory::Email),
        ];
        let resolved = resolve_overlaps(&mut spans);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].end, 10);
    }
}
