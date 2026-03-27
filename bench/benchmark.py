"""Benchmark unpii vs eds-pseudo on ai4privacy/pii-masking-300k dataset."""

from __future__ import annotations

import argparse
import time

from datasets import load_dataset

from metrics import EvalResult, MatchResult, check_leakage, match_spans, print_report

# ── Dataset label → unified category mapping ─────────────────────────────────

LABEL_MAP: dict[str, str] = {
    "GIVENNAME1": "PERSON",
    "GIVENNAME2": "PERSON",
    "LASTNAME1": "PERSON",
    "LASTNAME2": "PERSON",
    "LASTNAME3": "PERSON",
    "TITLE": "PERSON",
    "TEL": "PHONE",
    "EMAIL": "EMAIL",
    "DATE": "DATE",
    "TIME": "DATE",
    "BOD": "BIRTH_DATE",
    "ADDRESS": "LOCATION",
    "STREET": "LOCATION",
    "CITY": "LOCATION",
    "BUILDING": "LOCATION",
    "STATE": "LOCATION",
    "COUNTRY": "LOCATION",
    "POSTCODE": "ZIP_CODE",
    "SOCIALNUMBER": "SSN",
}

# eds-pseudo label → unified category
EDS_LABEL_MAP: dict[str, str] = {
    "NOM": "PERSON",
    "PRENOM": "PERSON",
    "TEL": "PHONE",
    "MAIL": "EMAIL",
    "DATE": "DATE",
    "DATE_NAISSANCE": "BIRTH_DATE",
    "ADRESSE": "LOCATION",
    "VILLE": "LOCATION",
    "ZIP": "ZIP_CODE",
    "SECU": "SSN",
    "IPP": "SSN",
    "NDA": "SSN",
    "HOPITAL": "LOCATION",
}


def prepare_ground_truth(privacy_mask: list[dict]) -> list[dict]:
    """Convert dataset annotations to ground truth spans, filtering unmapped categories."""
    gt = []
    for entry in privacy_mask:
        mapped = LABEL_MAP.get(entry["label"])
        if mapped is None:
            continue
        gt.append({
            "start": entry["start"],
            "end": entry["end"],
            "category": mapped,
            "value": entry["value"],
        })
    return gt


# ── Engine abstractions ──────────────────────────────────────────────────────

class UnpiiEngine:
    def __init__(self, mode: str = "standard"):
        import unpii
        self._unpii = unpii
        self.mode = mode
        self.name = f"unpii ({mode})"

    def find_spans(self, text: str) -> list[dict]:
        spans = self._unpii.find_spans(text, mode=self.mode)
        return [{"start": s.start, "end": s.end, "category": s.category} for s in spans]

    def anonymize(self, text: str) -> str:
        return self._unpii.anonymize(text, mode=self.mode)


class _EdsBaseEngine:
    """Shared logic for eds-pseudo engines (rules and ML)."""

    name: str
    nlp: object  # edsnlp pipeline

    def find_spans(self, text: str) -> list[dict]:
        doc = self.nlp(text)
        spans = []
        for ent in doc.ents:
            mapped = EDS_LABEL_MAP.get(ent.label_)
            if mapped is None:
                continue
            spans.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "category": mapped,
            })
        return spans

    def anonymize(self, text: str) -> str:
        doc = self.nlp(text)
        result = text
        for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
            mapped = EDS_LABEL_MAP.get(ent.label_)
            if mapped is None:
                continue
            result = result[: ent.start_char] + f"<{mapped}>" + result[ent.end_char :]
        return result


class EdsPseudoEngine(_EdsBaseEngine):
    def __init__(self):
        import edsnlp
        self.nlp = edsnlp.blank("eds")
        self.nlp.add_pipe("eds.normalizer")
        self.nlp.add_pipe(
            "eds_pseudo.simple_rules",
            config={"pattern_keys": ["TEL", "MAIL", "SECU", "PERSON"]},
        )
        self.nlp.add_pipe("eds_pseudo.addresses")
        self.nlp.add_pipe("eds_pseudo.dates")
        self.nlp.add_pipe("eds_pseudo.context")
        self.name = "eds-pseudo (rules)"


class EdsPseudoMLEngine(_EdsBaseEngine):
    def __init__(self):
        import edsnlp
        print("  Loading AP-HP/eds-pseudo-public model ...")
        self.nlp = edsnlp.load("AP-HP/eds-pseudo-public", auto_update=True)
        self.name = "eds-pseudo (ML)"


# ── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate(engine, docs: list[dict]) -> EvalResult:
    all_overlap: list[dict[str, MatchResult]] = []
    all_exact: list[dict[str, MatchResult]] = []
    all_leaks: list[list[dict]] = []
    total_pii = 0
    num_docs = len(docs)

    t0 = time.perf_counter()

    for i, row in enumerate(docs):
        text = row["source_text"]
        gt_spans = prepare_ground_truth(row["privacy_mask"])

        if not gt_spans:
            continue

        total_pii += len(gt_spans)

        predicted = engine.find_spans(text)

        overlap, exact = match_spans(predicted, gt_spans)
        all_overlap.append(overlap)
        all_exact.append(exact)

        masked = engine.anonymize(text)
        leaks = check_leakage(masked, gt_spans, doc_id=i)
        all_leaks.append(leaks)

        if (i + 1) % 500 == 0:
            elapsed_so_far = time.perf_counter() - t0
            print(f"  {i + 1}/{num_docs} docs ({elapsed_so_far:.1f}s) ...")

    elapsed = time.perf_counter() - t0

    result = EvalResult(total_pii=total_pii, elapsed_sec=elapsed, num_docs=num_docs)

    for doc_overlap in all_overlap:
        for cat, m in doc_overlap.items():
            if cat not in result.per_category:
                result.per_category[cat] = MatchResult()
            result.per_category[cat].tp += m.tp
            result.per_category[cat].fp += m.fp
            result.per_category[cat].fn += m.fn

    for doc_exact in all_exact:
        for cat, m in doc_exact.items():
            if cat not in result.per_category_exact:
                result.per_category_exact[cat] = MatchResult()
            result.per_category_exact[cat].tp += m.tp
            result.per_category_exact[cat].fp += m.fp
            result.per_category_exact[cat].fn += m.fn

    for doc_leaks in all_leaks:
        result.leaks.extend(doc_leaks)

    return result


def print_side_by_side(results: dict[str, EvalResult], num_docs: int) -> None:
    names = list(results.keys())
    col_w = 14

    print(f"\n{'=' * 60}")
    print(f"  Side-by-side comparison ({num_docs} docs)")
    print(f"{'=' * 60}\n")

    # ── Aggregate metrics ─────────────────────────────────────────────
    header = f"{'Metric':<20}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * len(header))

    micros = {n: r.micro for n, r in results.items()}
    for label, attr in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]:
        print(f"{label:<20}" + "".join(f"{getattr(micros[n], attr):>{col_w}.2%}" for n in names))

    micros_exact = {n: r.micro_exact for n, r in results.items()}
    print(f"{'Exact F1':<20}" + "".join(f"{micros_exact[n].f1:>{col_w}.2%}" for n in names))
    print(f"{'Leak rate':<20}" + "".join(f"{results[n].leak_rate:>{col_w}.2%}" for n in names))
    print(f"{'Throughput':<20}" + "".join(f"{results[n].throughput:>{col_w - 2}.0f}/s" for n in names))
    print()

    # ── Per-category breakdown ────────────────────────────────────────
    all_cats = sorted(set().union(*(r.per_category.keys() for r in results.values())))
    cat_header = f"{'Category':<14} {'Count':>6}"
    for n in names:
        cat_header += f"  {'Prec':>6} {'Rec':>6} {'Leak':>5}"
    label_row = f"{'':>21}"
    for n in names:
        label_row += f"  {'── ' + n + ' ──':>19}"
    print(label_row)
    print(cat_header)
    print("-" * len(cat_header))

    leak_totals = {n: 0 for n in names}
    for cat in all_cats:
        # Use first engine that has the category for count
        first_m = next((results[n].per_category.get(cat) for n in names if cat in results[n].per_category), MatchResult())
        count = first_m.tp + first_m.fn
        row = f"{cat:<14} {count:>6}"
        for n in names:
            m = results[n].per_category.get(cat, MatchResult())
            leaks = sum(1 for l in results[n].leaks if l["category"] == cat)
            leak_totals[n] += leaks
            row += f"  {m.precision:>5.0%} {m.recall:>6.0%} {leaks:>5}"
        print(row)

    print("-" * len(cat_header))
    first_result = next(iter(results.values()))
    row = f"{'TOTAL':<14} {first_result.total_pii:>6}"
    for n in names:
        micro = results[n].micro
        row += f"  {micro.precision:>5.0%} {micro.recall:>6.0%} {leak_totals[n]:>5}"
    print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark unpii vs eds-pseudo")
    parser.add_argument("--mode", default="paranoid", choices=["standard", "paranoid"])
    parser.add_argument("--limit", type=int, default=None, help="Max French docs to evaluate")
    parser.add_argument(
        "--engine", default="all",
        choices=["all", "unpii", "eds-pseudo", "eds-pseudo-ml"],
        help="Which engine(s) to benchmark",
    )
    args = parser.parse_args()

    print("Loading dataset ai4privacy/pii-masking-300k ...")
    ds = load_dataset("ai4privacy/pii-masking-300k", split="train")

    print("Filtering French documents ...")
    ds_fr = ds.filter(lambda x: x["language"] == "French")

    if args.limit:
        ds_fr = ds_fr.select(range(min(args.limit, len(ds_fr))))

    docs = list(ds_fr)
    print(f"Evaluating on {len(docs)} French documents\n")

    results: dict[str, EvalResult] = {}
    engines_to_run: list[str] = (
        ["unpii", "eds-pseudo", "eds-pseudo-ml"] if args.engine == "all"
        else [args.engine]
    )

    # ── unpii ─────────────────────────────────────────────────────────────
    if "unpii" in engines_to_run:
        print(f"Running unpii ({args.mode}) ...")
        engine = UnpiiEngine(mode=args.mode)
        results["unpii"] = evaluate(engine, docs)
        print_report(results["unpii"], engine.name)

    # ── eds-pseudo (rules) ────────────────────────────────────────────────
    if "eds-pseudo" in engines_to_run:
        print("Running eds-pseudo (rules) ...")
        engine = EdsPseudoEngine()
        results["eds-pseudo"] = evaluate(engine, docs)
        print_report(results["eds-pseudo"], engine.name)

    # ── eds-pseudo (ML) ──────────────────────────────────────────────────
    if "eds-pseudo-ml" in engines_to_run:
        print("Running eds-pseudo (ML) ...")
        engine = EdsPseudoMLEngine()
        results["eds-pseudo-ml"] = evaluate(engine, docs)
        print_report(results["eds-pseudo-ml"], engine.name)

    # ── Side-by-side summary ──────────────────────────────────────────────
    if len(results) >= 2:
        print_side_by_side(results, len(docs))

    # ── Parallel throughput benchmark ─────────────────────────────────────
    run_throughput_benchmark(docs, args.mode, engines_to_run, results)


def run_throughput_benchmark(
    docs: list[dict],
    mode: str,
    engines: list[str],
    results: dict[str, EvalResult],
) -> None:
    import os

    texts = [row["source_text"] for row in docs]
    num_docs = len(texts)
    num_cores = os.cpu_count() or 1

    print(f"{'=' * 60}")
    print(f"  Throughput benchmark (parallel, {num_cores} cores)")
    print(f"{'=' * 60}\n")

    parallel: dict[str, float] = {}

    if "unpii" in engines:
        import unpii
        t0 = time.perf_counter()
        unpii.anonymize_batch(texts, mode=mode)
        elapsed = time.perf_counter() - t0
        parallel["unpii"] = num_docs / elapsed if elapsed > 0 else 0

    if "eds-pseudo" in engines:
        import edsnlp
        nlp = edsnlp.blank("eds")
        nlp.add_pipe("eds.normalizer")
        nlp.add_pipe(
            "eds_pseudo.simple_rules",
            config={"pattern_keys": ["TEL", "MAIL", "SECU", "PERSON"]},
        )
        nlp.add_pipe("eds_pseudo.addresses")
        nlp.add_pipe("eds_pseudo.dates")
        nlp.add_pipe("eds_pseudo.context")

        t0 = time.perf_counter()
        stream = nlp.pipe(texts)
        stream = stream.set_processing(num_cpu_workers=num_cores, batch_size=64)
        for _ in stream:
            pass
        elapsed = time.perf_counter() - t0
        parallel["eds-pseudo"] = num_docs / elapsed if elapsed > 0 else 0

    if "eds-pseudo-ml" in engines:
        import edsnlp
        nlp = edsnlp.load("AP-HP/eds-pseudo-public", auto_update=True)

        t0 = time.perf_counter()
        stream = nlp.pipe(texts)
        stream = stream.set_processing(num_cpu_workers=num_cores, batch_size=64)
        for _ in stream:
            pass
        elapsed = time.perf_counter() - t0
        parallel["eds-pseudo-ml"] = num_docs / elapsed if elapsed > 0 else 0

    col_w = 14
    names = [n for n in engines if n in parallel]
    header = f"{'':20s}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * len(header))

    print(f"{'Single-thread':<20}" + "".join(
        f"{results[n].throughput:>{col_w - 2}.0f}/s" if n in results else f"{'—':>{col_w}}"
        for n in names
    ))
    print(f"{'Parallel':<20}" + "".join(
        f"{parallel[n]:>{col_w - 2}.0f}/s" for n in names
    ))
    print(f"{'Speedup':<20}" + "".join(
        f"{parallel[n] / results[n].throughput:>{col_w - 2}.1f}x"
        if n in results and results[n].throughput > 0 else f"{'—':>{col_w}}"
        for n in names
    ))
    print()


if __name__ == "__main__":
    main()
