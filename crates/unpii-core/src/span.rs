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
/// When same length, prefer the more specific category (lower priority value).
pub fn resolve_overlaps(spans: &mut Vec<Span>) -> Vec<Span> {
    if spans.is_empty() {
        return Vec::new();
    }

    // Sort by: start asc, length desc, priority asc (more specific wins)
    spans.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(b.len().cmp(&a.len()))
            .then(a.category.priority().cmp(&b.category.priority()))
    });

    let mut result: Vec<Span> = Vec::new();
    for span in spans.drain(..) {
        if let Some(last) = result.last_mut() {
            if span.start < last.end {
                // Overlap: if same extent but new span has higher priority, replace
                if span.start == last.start && span.len() == last.len()
                    && span.category.priority() < last.category.priority()
                {
                    *last = span;
                }
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
            Span::new(0, 5, PiiCategory::Person),
            Span::new(10, 15, PiiCategory::Email),
        ];
        let resolved = resolve_overlaps(&mut spans);
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn test_overlap_keeps_longest() {
        let mut spans = vec![
            Span::new(0, 10, PiiCategory::Person),
            Span::new(5, 8, PiiCategory::Email),
        ];
        let resolved = resolve_overlaps(&mut spans);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].end, 10);
    }
}
