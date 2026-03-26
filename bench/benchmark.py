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
    "BOD": "BIRTHDATE",
    "ADDRESS": "LOCATION",
    "STREET": "LOCATION",
    "CITY": "LOCATION",
    "BUILDING": "LOCATION",
    "STATE": "LOCATION",
    "COUNTRY": "LOCATION",
    "POSTCODE": "ZIP_CODE",
    "SOCIALNUMBER": "NIR",
}

# eds-pseudo label → unified category
EDS_LABEL_MAP: dict[str, str] = {
    "NOM": "PERSON",
    "PRENOM": "PERSON",
    "TEL": "PHONE",
    "MAIL": "EMAIL",
    "DATE": "DATE",
    "DATE_NAISSANCE": "BIRTHDATE",
    "ADRESSE": "LOCATION",
    "VILLE": "LOCATION",
    "ZIP": "ZIP_CODE",
    "SECU": "NIR",
    "IPP": "NIR",
    "NDA": "NIR",
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


class EdsPseudoEngine:
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark unpii vs eds-pseudo")
    parser.add_argument("--mode", default="paranoid", choices=["standard", "paranoid"])
    parser.add_argument("--limit", type=int, default=None, help="Max French docs to evaluate")
    parser.add_argument("--engine", default="all", choices=["all", "unpii", "eds-pseudo"],
                        help="Which engine(s) to benchmark")
    args = parser.parse_args()

    print("Loading dataset ai4privacy/pii-masking-300k ...")
    ds = load_dataset("ai4privacy/pii-masking-300k", split="train")

    print("Filtering French documents ...")
    ds_fr = ds.filter(lambda x: x["language"] == "French")

    if args.limit:
        ds_fr = ds_fr.select(range(min(args.limit, len(ds_fr))))

    docs = list(ds_fr)
    print(f"Evaluating on {len(docs)} French documents\n")

    unpii_result = None
    eds_result = None

    # ── unpii ─────────────────────────────────────────────────────────────
    if args.engine in ("all", "unpii"):
        print(f"Running unpii ({args.mode}) ...")
        unpii_engine = UnpiiEngine(mode=args.mode)
        unpii_result = evaluate(unpii_engine, docs)
        print_report(unpii_result, unpii_engine.name)

    # ── eds-pseudo ────────────────────────────────────────────────────────
    if args.engine in ("all", "eds-pseudo"):
        print("Running eds-pseudo (rules) ...")
        eds_engine = EdsPseudoEngine()
        eds_result = evaluate(eds_engine, docs)
        print_report(eds_result, eds_engine.name)

    # ── Side-by-side summary ──────────────────────────────────────────────
    if unpii_result and eds_result:
        print(f"\n{'=' * 60}")
        print(f"  Side-by-side comparison ({len(docs)} docs)")
        print(f"{'=' * 60}\n")

        header = f"{'Metric':<20} {'unpii':>12} {'eds-pseudo':>12}"
        print(header)
        print("-" * len(header))

        u, e = unpii_result.micro, eds_result.micro
        print(f"{'Precision':<20} {u.precision:>11.2%} {e.precision:>12.2%}")
        print(f"{'Recall':<20} {u.recall:>11.2%} {e.recall:>12.2%}")
        print(f"{'F1':<20} {u.f1:>11.2%} {e.f1:>12.2%}")

        ue, ee = unpii_result.micro_exact, eds_result.micro_exact
        print(f"{'Exact F1':<20} {ue.f1:>11.2%} {ee.f1:>12.2%}")

        print(f"{'Leak rate':<20} {unpii_result.leak_rate:>11.2%} {eds_result.leak_rate:>12.2%}")
        print(f"{'Throughput':<20} {unpii_result.throughput:>10.0f}/s {eds_result.throughput:>10.0f}/s")
        print()

        # Per-category side-by-side
        all_cats = sorted(
            set(unpii_result.per_category.keys()) | set(eds_result.per_category.keys())
        )
        header2 = f"{'Category':<14} {'Count':>6}  {'Prec':>6} {'Rec':>6} {'Leak':>5}  {'Prec':>6} {'Rec':>6} {'Leak':>5}"
        print(f"{'':20s} {'── unpii ──':>19}  {'── eds-pseudo ──':>19}")
        print(header2)
        print("-" * len(header2))
        u_total, e_total = 0, 0
        for cat in all_cats:
            um = unpii_result.per_category.get(cat, MatchResult())
            em = eds_result.per_category.get(cat, MatchResult())
            count = um.tp + um.fn
            u_leaks = sum(1 for l in unpii_result.leaks if l["category"] == cat)
            e_leaks = sum(1 for l in eds_result.leaks if l["category"] == cat)
            u_total += u_leaks
            e_total += e_leaks
            print(
                f"{cat:<14} {count:>6}  {um.precision:>5.0%} {um.recall:>6.0%} {u_leaks:>5}"
                f"  {em.precision:>5.0%} {em.recall:>6.0%} {e_leaks:>5}"
            )
        u_micro, e_micro = unpii_result.micro, eds_result.micro
        print("-" * len(header2))
        print(
            f"{'TOTAL':<14} {unpii_result.total_pii:>6}  {u_micro.precision:>5.0%} {u_micro.recall:>6.0%} {u_total:>5}"
            f"  {e_micro.precision:>5.0%} {e_micro.recall:>6.0%} {e_total:>5}"
        )
        print()

    # ── Parallel throughput benchmark ─────────────────────────────────────
    run_throughput_benchmark(
        docs, args.mode, args.engine,
        unpii_result.throughput if unpii_result else 0,
        eds_result.throughput if eds_result else 0,
    )


def run_throughput_benchmark(
    docs: list[dict],
    mode: str,
    engine_filter: str,
    unpii_st: float,
    eds_st: float,
) -> None:
    import os

    texts = [row["source_text"] for row in docs]
    num_docs = len(texts)
    num_cores = os.cpu_count() or 1

    print(f"{'=' * 60}")
    print(f"  Throughput benchmark (parallel, {num_cores} cores)")
    print(f"{'=' * 60}\n")

    unpii_par = 0.0
    eds_par = 0.0

    if engine_filter in ("all", "unpii"):
        import unpii
        t0 = time.perf_counter()
        unpii.anonymize_batch(texts, mode=mode)
        unpii_elapsed = time.perf_counter() - t0
        unpii_par = num_docs / unpii_elapsed if unpii_elapsed > 0 else 0

    if engine_filter in ("all", "eds-pseudo"):
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
        eds_elapsed = time.perf_counter() - t0
        eds_par = num_docs / eds_elapsed if eds_elapsed > 0 else 0

    if engine_filter == "all":
        header = f"{'':20s} {'unpii':>12} {'eds-pseudo':>12}"
        print(header)
        print("-" * len(header))
        print(f"{'Single-thread':<20} {unpii_st:>10.0f}/s {eds_st:>10.0f}/s")
        print(f"{'Parallel':<20} {unpii_par:>10.0f}/s {eds_par:>10.0f}/s")
        unpii_speedup = unpii_par / unpii_st if unpii_st > 0 else 0
        eds_speedup = eds_par / eds_st if eds_st > 0 else 0
        print(f"{'Speedup':<20} {unpii_speedup:>10.1f}x {eds_speedup:>10.1f}x")
    else:
        par = unpii_par if engine_filter == "unpii" else eds_par
        st = unpii_st if engine_filter == "unpii" else eds_st
        speedup = par / st if st > 0 else 0
        print(f"Single-thread: {st:,.0f}/s")
        print(f"Parallel:      {par:,.0f}/s")
        print(f"Speedup:       {speedup:.1f}x")
    print()


if __name__ == "__main__":
    main()
