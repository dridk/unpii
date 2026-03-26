"""Evaluation metrics for unpii anonymization benchmark."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class MatchResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class EvalResult:
    per_category: dict[str, MatchResult] = field(default_factory=dict)
    per_category_exact: dict[str, MatchResult] = field(default_factory=dict)
    leaks: list[dict] = field(default_factory=list)
    total_pii: int = 0
    elapsed_sec: float = 0.0
    num_docs: int = 0

    @property
    def micro(self) -> MatchResult:
        r = MatchResult()
        for m in self.per_category.values():
            r.tp += m.tp
            r.fp += m.fp
            r.fn += m.fn
        return r

    @property
    def micro_exact(self) -> MatchResult:
        r = MatchResult()
        for m in self.per_category_exact.values():
            r.tp += m.tp
            r.fp += m.fp
            r.fn += m.fn
        return r

    @property
    def leak_rate(self) -> float:
        return len(self.leaks) / self.total_pii if self.total_pii > 0 else 0.0

    @property
    def throughput(self) -> float:
        return self.num_docs / self.elapsed_sec if self.elapsed_sec > 0 else 0.0


def _iou(s1_start: int, s1_end: int, s2_start: int, s2_end: int) -> float:
    inter_start = max(s1_start, s2_start)
    inter_end = min(s1_end, s2_end)
    intersection = max(0, inter_end - inter_start)
    union = (s1_end - s1_start) + (s2_end - s2_start) - intersection
    return intersection / union if union > 0 else 0.0


def match_spans(
    predicted: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[dict[str, MatchResult], dict[str, MatchResult]]:
    """Match predicted spans to ground truth.

    Returns (overlap_results, exact_results) per category.
    """
    overlap_results: dict[str, MatchResult] = defaultdict(MatchResult)
    exact_results: dict[str, MatchResult] = defaultdict(MatchResult)

    used_pred = set()

    for gt in ground_truth:
        cat = gt["category"]
        best_pred_idx = None
        best_iou = 0.0
        exact_match = False

        for i, pred in enumerate(predicted):
            if i in used_pred:
                continue
            if pred["category"] != cat:
                continue
            iou = _iou(pred["start"], pred["end"], gt["start"], gt["end"])
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = i
                exact_match = pred["start"] == gt["start"] and pred["end"] == gt["end"]

        if best_pred_idx is not None and best_iou >= iou_threshold:
            used_pred.add(best_pred_idx)
            overlap_results[cat].tp += 1
            if exact_match:
                exact_results[cat].tp += 1
            else:
                exact_results[cat].fn += 1
        else:
            overlap_results[cat].fn += 1
            exact_results[cat].fn += 1

    for i, pred in enumerate(predicted):
        if i not in used_pred:
            overlap_results[pred["category"]].fp += 1
            exact_results[pred["category"]].fp += 1

    all_cats = {gt["category"] for gt in ground_truth} | {p["category"] for p in predicted}
    for cat in all_cats:
        if cat not in overlap_results:
            overlap_results[cat] = MatchResult()
        if cat not in exact_results:
            exact_results[cat] = MatchResult()

    return dict(overlap_results), dict(exact_results)


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[\s\-\.\,\/\(\)\+]", "", s)
    return s.lower()


def check_leakage(
    masked_text: str,
    ground_truth_spans: list[dict],
    doc_id: int = 0,
) -> list[dict]:
    """Check if any PII values still appear in the masked text."""
    leaks = []
    masked_lower = masked_text.lower()
    masked_normalized = _normalize(masked_text)

    for gt in ground_truth_spans:
        value = gt["value"]
        cat = gt["category"]

        if len(value.strip()) < 3:
            continue

        if value.lower() in masked_lower:
            leaks.append({"doc_id": doc_id, "category": cat, "value": value, "type": "exact"})
            continue

        if cat == "NOM":
            for part in value.split():
                if part.lower().rstrip(".") in {
                    "dr", "docteur", "pr", "professeur", "m", "mme", "mr",
                }:
                    continue
                if len(part) >= 4 and part.lower() in masked_lower:
                    leaks.append({
                        "doc_id": doc_id, "category": cat,
                        "value": f"{value} (component: {part})", "type": "component",
                    })
                    break
        else:
            norm_val = _normalize(value)
            if norm_val and len(norm_val) >= 4 and norm_val in masked_normalized:
                leaks.append({"doc_id": doc_id, "category": cat, "value": value, "type": "normalized"})

    return leaks


def print_report(result: EvalResult, mode: str) -> None:
    """Print evaluation report."""
    print(f"\n{'=' * 78}")
    print(f"  unpii Benchmark Report")
    print(f"  Mode: {mode} | Docs: {result.num_docs} | PII spans: {result.total_pii}")
    print(f"  Throughput: {result.throughput:,.0f} docs/sec ({result.elapsed_sec:.2f}s)")
    print(f"{'=' * 78}\n")

    cats = sorted(result.per_category.keys())
    header = f"{'Category':<14} {'Count':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ExactF1':>8} {'Leaked':>7}"
    print(header)
    print("-" * len(header))

    for cat in cats:
        m = result.per_category.get(cat, MatchResult())
        me = result.per_category_exact.get(cat, MatchResult())
        leak_count = sum(1 for l in result.leaks if l["category"] == cat)
        count = m.tp + m.fn
        print(
            f"{cat:<14} {count:>6} {m.precision:>7.2%} {m.recall:>7.2%} "
            f"{m.f1:>7.2%} {me.f1:>8.2%} {leak_count:>7}"
        )

    micro = result.micro
    micro_exact = result.micro_exact
    total_leaks = len(result.leaks)
    print("-" * len(header))
    print(
        f"{'MICRO':<14} {micro.tp + micro.fn:>6} {micro.precision:>7.2%} {micro.recall:>7.2%} "
        f"{micro.f1:>7.2%} {micro_exact.f1:>8.2%} {total_leaks:>7}"
    )

    print(f"\nLeak rate: {total_leaks}/{result.total_pii} ({result.leak_rate:.2%})")
    print()
