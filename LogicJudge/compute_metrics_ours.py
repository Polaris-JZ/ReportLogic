import argparse
import csv
import glob
import json
import os
import re
import warnings
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_GT_COLUMN = "总体获胜方"
DEFAULT_PRED_COLUMN = "overall_winner"
DEFAULT_GT_ENCODING = "utf-8"
DEFAULT_PRED_ENCODING = "utf-8"
DEFAULT_FALLBACK_ENCODINGS = ["utf-8-sig", "gb18030", "gbk"]

MISSING_PREDICTION_TOKEN = "__missing_prediction__"
MISSING_GROUND_TRUTH_TOKEN = "__missing_ground_truth__"


def _read_csv_rows(path: str, encoding: str, enable_fallback: bool) -> List[Dict[str, Any]]:
    encodings: List[str] = [encoding]
    if enable_fallback:
        seen = {encoding.lower()}
        for candidate in DEFAULT_FALLBACK_ENCODINGS:
            lowered = candidate.lower()
            if lowered not in seen:
                encodings.append(candidate)
                seen.add(lowered)

    last_error: Optional[Exception] = None
    for candidate in encodings:
        try:
            with open(path, "r", encoding=candidate, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [row for row in reader]
            if candidate != encoding:
                warnings.warn(
                    f"Decoded CSV '{path}' using fallback encoding '{candidate}'",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return rows
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(
        "Failed to decode CSV file using encodings: "
        + ", ".join(encodings)
        + (f". Last error: {last_error}" if last_error else "")
    )


def _read_jsonl_rows(path: str, encoding: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding=encoding) as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON object at line {line_number} in '{path}': {exc}"
                ) from exc
    return rows


def _detect_loader(path: str) -> str:
    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        return "csv"
    if ext in {".jsonl", ".json"}:
        return "jsonl"
    raise ValueError(
        f"Unsupported file extension '{ext}' for '{path}'. Expected .csv or .jsonl/.json"
    )


def _normalize_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _parse_label_map(entries: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in entries:
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"Invalid label map entry '{entry}'. Expected format 'source=target'."
            )
        source, target = entry.split("=", 1)
        source_norm = _normalize_label(source)
        target_norm = _normalize_label(target)
        if not source_norm:
            raise ValueError(f"Label map entry '{entry}' has an empty source value after normalization.")
        if not target_norm:
            raise ValueError(f"Label map entry '{entry}' has an empty target value after normalization.")
        mapping[source_norm] = target_norm
    return mapping


def _make_label_transform(mapping: Dict[str, str]) -> Optional[Callable[[str], str]]:
    if not mapping:
        return None

    def _transform(label: str) -> str:
        normalized = _normalize_label(label)
        if not normalized:
            return normalized
        return mapping.get(normalized, normalized)

    return _transform


def _extract_label_from_payload(payload: Any, key: str) -> Optional[str]:
    if payload is None or not key:
        return None

    def _search(obj: Any) -> Optional[str]:
        if isinstance(obj, dict):
            if key in obj:
                value = _normalize_label(obj.get(key))
                if value:
                    return value
            for child in obj.values():
                found = _search(child)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _search(item)
                if found:
                    return found
        return None

    if isinstance(payload, dict):
        return _search(payload)

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None

        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        candidate_texts: List[str] = []
        if cleaned:
            candidate_texts.append(cleaned)
        if cleaned != text:
            candidate_texts.append(text)
        if not candidate_texts:
            candidate_texts.append(text)

        decoder = json.JSONDecoder()
        seen_candidates = set()

        for candidate in candidate_texts:
            candidate = candidate.strip()
            if not candidate:
                continue
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)

            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None

            if parsed is not None:
                found = _search(parsed)
                if found:
                    return found

            index = 0
            length = len(candidate)
            while index < length:
                try:
                    parsed_obj, end_index = decoder.raw_decode(candidate, idx=index)
                except json.JSONDecodeError:
                    index += 1
                    continue
                found = _search(parsed_obj)
                if found:
                    return found
                if end_index <= index:
                    index += 1
                else:
                    index = end_index

            snippets = re.split(r"(?<=\})(?=\s*\{)", candidate)
            for snippet in reversed(snippets):
                snippet = snippet.strip()
                if not snippet or snippet in seen_candidates:
                    continue
                try:
                    parsed_obj = json.loads(snippet)
                except json.JSONDecodeError:
                    continue
                found = _search(parsed_obj)
                if found:
                    return found

        fallback_patterns = [
            (rf'"{re.escape(key)}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', True),
            (rf"'{re.escape(key)}'\s*:\s*'([^'\\]*(?:\\.[^'\\]*)*)'", True),
            (rf"{re.escape(key)}\s*[:=]\s*([A-Za-z])", False),
        ]
        for pattern, is_quoted in fallback_patterns:
            for candidate in candidate_texts:
                match = re.search(pattern, candidate)
                if not match:
                    continue
                raw_value = match.group(1)
                if is_quoted:
                    try:
                        value = json.loads(f'"{raw_value}"')
                    except json.JSONDecodeError:
                        value = raw_value
                else:
                    value = raw_value
                normalized = _normalize_label(value)
                if normalized:
                    return normalized
        return None

    return None


def _make_auto_id_fn(prefix_length: int) -> Callable[[Dict[str, Any]], Optional[str]]:
    if prefix_length < 0:
        raise ValueError("prefix_length must be non-negative")

    question_keys = (
        "question_id",
        "query_id",
        "sample_id",
        "id",
        "query",
        "question",
    )
    answer_a_keys = ("answer_a", "text_a", "response_a", "assistant_1")
    answer_b_keys = ("answer_b", "text_b", "response_b", "assistant_2")

    def _derive(row: Dict[str, Any]) -> Optional[str]:
        def _pick(keys: Sequence[str]) -> Optional[str]:
            for key in keys:
                if key in row:
                    value = _normalize_label(row.get(key))
                    if value:
                        return value
            return None

        question_part = _pick(question_keys)
        answer_a_part = _pick(answer_a_keys)
        answer_b_part = _pick(answer_b_keys)

        if question_part is None and answer_a_part is None and answer_b_part is None:
            return None

        def _clip(value: Optional[str]) -> str:
            if value is None:
                return ""
            return value[:prefix_length] if prefix_length and len(value) > prefix_length else value

        combined = "||".join([
            _clip(question_part),
            _clip(answer_a_part),
            _clip(answer_b_part),
        ])

        combined = combined.strip("|")
        return combined or None

    return _derive


def _load_labels(
    path: str,
    label_column: str,
    id_column: Optional[str],
    encoding: str,
    enable_fallback: bool,
    fallback_json_sources: Optional[Sequence[Tuple[str, str]]] = None,
    derive_id_fn: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    label_transform: Optional[Callable[[str], str]] = None,
) -> List[Tuple[str, str]]:
    loader = _detect_loader(path)
    if loader == "csv":
        rows = _read_csv_rows(path, encoding, enable_fallback)
    else:
        rows = _read_jsonl_rows(path, encoding)

    records: List[Tuple[str, str]] = []
    skipped = 0
    for index, row in enumerate(rows):
        if row is None:
            continue
        raw_label = row.get(label_column)
        label = _normalize_label(raw_label)

        if (not label) and fallback_json_sources:
            for field_name, json_key in fallback_json_sources:
                candidate = _extract_label_from_payload(row.get(field_name), json_key)
                if candidate:
                    label = candidate
                    break

        if label and label_transform:
            label = _normalize_label(label_transform(label))

        if not label:
            reason = f"label column '{label_column}' is empty"
            if fallback_json_sources:
                reason += " and no fallback succeeded"
            if label_transform:
                reason += " (after applying label mapping)"
            warnings.warn(
                f"Skipping row {index} in '{path}' because {reason}.",
                RuntimeWarning,
                stacklevel=2,
            )
            skipped += 1
            continue
        if id_column:
            if id_column not in row:
                raise KeyError(
                    f"ID column '{id_column}' not found in file '{path}'. Available keys: {sorted(row.keys())}"
                )
            identifier = _normalize_label(row[id_column])
            if not identifier:
                raise ValueError(
                    f"Empty identifier for column '{id_column}' at index {index} in '{path}'"
                )
        else:
            if derive_id_fn:
                identifier = derive_id_fn(row)
                if not identifier:
                    warnings.warn(
                        f"Skipping row {index} in '{path}' because automatic ID derivation failed.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    skipped += 1
                    continue
            else:
                identifier = str(index)
        records.append((identifier, label))
    if not records:
        raise RuntimeError(f"No valid labels found in '{path}'. Skipped rows: {skipped}")
    if skipped:
        warnings.warn(
            f"Skipped {skipped} row(s) with empty labels in '{path}'.",
            RuntimeWarning,
            stacklevel=2,
        )
    return records


def _load_combined_labels(
    path: str,
    pred_field: str,
    pred_key: str,
    gt_field: str,
    gt_key: str,
    encoding: str,
    pred_transform: Optional[Callable[[str], str]] = None,
    gt_transform: Optional[Callable[[str], str]] = None,
) -> Tuple[List[str], List[str]]:
    rows = _read_jsonl_rows(path, encoding)

    gt_labels: List[str] = []
    pred_labels: List[str] = []
    penalized = 0

    for index, row in enumerate(rows):
        if row is None:
            continue

        pred_label = _extract_label_from_payload(row.get(pred_field), pred_key)
        gt_label = _extract_label_from_payload(row.get(gt_field), gt_key)

        if pred_label and pred_transform:
            pred_label = pred_transform(pred_label)
        if gt_label and gt_transform:
            gt_label = gt_transform(gt_label)

        normalized_pred = _normalize_label(pred_label)
        normalized_gt = _normalize_label(gt_label)

        missing_pred = not normalized_pred
        missing_gt = not normalized_gt

        if missing_pred or missing_gt:
            penalized += 1
            warnings.warn(
                (
                    f"Penalizing row {index + 1} in '{path}' because "
                    f"{'prediction is missing' if missing_pred else ''}"
                    f"{' and ' if missing_pred and missing_gt else ''}"
                    f"{'ground truth is missing' if missing_gt else ''}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            if missing_pred:
                normalized_pred = MISSING_PREDICTION_TOKEN
            if missing_gt:
                normalized_gt = MISSING_GROUND_TRUTH_TOKEN

        pred_labels.append(normalized_pred)
        gt_labels.append(normalized_gt)

    if not pred_labels:
        raise RuntimeError(f"No valid label pairs found in '{path}'.")

    if penalized:
        warnings.warn(
            f"Penalized {penalized} row(s) with missing prediction or ground truth in '{path}'.",
            RuntimeWarning,
            stacklevel=2,
        )

    return gt_labels, pred_labels


def _align_labels(
    gt_records: List[Tuple[str, str]],
    pred_records: List[Tuple[str, str]],
    use_ids: bool,
) -> Tuple[List[str], List[str]]:
    if use_ids:
        gt_map = {identifier: label for identifier, label in gt_records}

        missing_in_pred = sorted(set(gt_map) - {identifier for identifier, _ in pred_records})
        if missing_in_pred:
            warnings.warn(
                f"{len(missing_in_pred)} ground-truth IDs are missing in predictions. They will be ignored: {missing_in_pred[:5]}"
                + ("..." if len(missing_in_pred) > 5 else ""),
                RuntimeWarning,
                stacklevel=2,
            )

        missing_in_gt = []
        aligned_gt: List[str] = []
        aligned_pred: List[str] = []
        for identifier, pred_label in pred_records:
            if identifier not in gt_map:
                missing_in_gt.append(identifier)
                continue
            aligned_gt.append(gt_map[identifier])
            aligned_pred.append(pred_label)

        if missing_in_gt:
            warnings.warn(
                f"{len(missing_in_gt)} prediction IDs are missing in ground truth. They were skipped: {missing_in_gt[:5]}"
                + ("..." if len(missing_in_gt) > 5 else ""),
                RuntimeWarning,
                stacklevel=2,
            )

        if not aligned_pred:
            raise RuntimeError("No overlapping IDs between ground truth and predictions.")

        return aligned_gt, aligned_pred

    if len(gt_records) != len(pred_records):
        raise RuntimeError(
            "Ground truth and prediction files must contain the same number of rows when no ID column is provided. "
            f"Got {len(gt_records)} vs {len(pred_records)}"
        )
    gt_labels = [label for _, label in gt_records]
    pred_labels = [label for _, label in pred_records]
    return gt_labels, pred_labels


def _compute_metrics(gt_labels: Sequence[str], pred_labels: Sequence[str]) -> Dict[str, Any]:
    if len(gt_labels) != len(pred_labels):
        raise ValueError("Mismatched label sequences; cannot compute metrics.")

    total = len(gt_labels)
    labels = sorted(set(gt_labels) | set(pred_labels))
    support = Counter(gt_labels)

    results: Dict[str, Any] = {
        "total": total,
        "labels": {},
    }

    correct = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
    agreement = correct / total if total else 0.0
    results["agreement"] = agreement

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in labels:
        tp = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == label and pred == label)
        fp = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt != label and pred == label)
        fn = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        results["labels"][label] = {
            "support": support[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    label_count = len(labels)
    if label_count:
        results["macro_precision"] = macro_precision / label_count
        results["macro_recall"] = macro_recall / label_count
        results["macro_f1"] = macro_f1 / label_count
    else:
        results["macro_precision"] = 0.0
        results["macro_recall"] = 0.0
        results["macro_f1"] = 0.0

    # Micro-averaged metrics are equivalent to accuracy for multiclass single-label problems
    results["micro_precision"] = agreement
    results["micro_recall"] = agreement
    results["micro_f1"] = agreement

    return results


def _format_percentage(value: float) -> str:
    return f"{value * 100:6.2f}%"


def _print_report(metrics: Dict[str, Any]) -> None:
    total = metrics["total"]
    agreement = metrics["agreement"]
    print(f"Total samples evaluated: {total}")
    print(f"Agreement (accuracy): {_format_percentage(agreement)}")
    print()
    print(
        f"{'Label':<20}{'Support':>10}{'Precision':>12}{'Recall':>12}{'F1':>12}{'TP':>8}{'FP':>8}{'FN':>8}"
    )
    print("-" * 82)
    for label, stats in sorted(
        metrics["labels"].items(), key=lambda item: (-item[1]["support"], item[0])
    ):
        print(
            f"{label:<20}{stats['support']:>10d}{_format_percentage(stats['precision']):>12}"
            f"{_format_percentage(stats['recall']):>12}{_format_percentage(stats['f1']):>12}"
            f"{stats['tp']:>8d}{stats['fp']:>8d}{stats['fn']:>8d}"
        )
    print("-" * 82)
    print(
        f"{'Macro Avg':<20}{'':>10}{_format_percentage(metrics['macro_precision']):>12}"
        f"{_format_percentage(metrics['macro_recall']):>12}{_format_percentage(metrics['macro_f1']):>12}"
        f"{'':>8}{'':>8}{'':>8}"
    )
    print(
        f"{'Micro Avg':<20}{'':>10}{_format_percentage(metrics['micro_precision']):>12}"
        f"{_format_percentage(metrics['micro_recall']):>12}{_format_percentage(metrics['micro_f1']):>12}"
        f"{'':>8}{'':>8}{'':>8}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute agreement, precision, recall, and F1 between ground-truth winners and model predictions.",
    )
    parser.add_argument("--gt", help="Path to ground-truth CSV/JSONL file")
    parser.add_argument(
        "--pred",
        nargs="+",
        help="One or more prediction CSV/JSONL files or directories containing JSONL files",
    )
    parser.add_argument(
        "--combined",
        help="Optional path to a JSONL file containing both ground-truth and predictions",
    )
    parser.add_argument(
        "--gt-column",
        default=DEFAULT_GT_COLUMN,
        help=f"Column name containing ground-truth labels (default: {DEFAULT_GT_COLUMN})",
    )
    parser.add_argument(
        "--pred-column",
        default=DEFAULT_PRED_COLUMN,
        help=f"Column name containing predicted labels (default: {DEFAULT_PRED_COLUMN})",
    )
    parser.add_argument(
        "--gt-id-column",
        default=None,
        help="Optional column name used as unique identifier in the ground-truth file",
    )
    parser.add_argument(
        "--pred-id-column",
        default=None,
        help="Optional column name used as unique identifier in the prediction file",
    )
    parser.add_argument(
        "--gt-encoding",
        default=DEFAULT_GT_ENCODING,
        help=f"Text encoding for ground-truth file (default: {DEFAULT_GT_ENCODING})",
    )
    parser.add_argument(
        "--pred-encoding",
        default=DEFAULT_PRED_ENCODING,
        help=f"Text encoding for prediction file (default: {DEFAULT_PRED_ENCODING})",
    )
    parser.add_argument(
        "--combined-encoding",
        default=DEFAULT_PRED_ENCODING,
        help="Text encoding for combined JSONL file when --combined is provided",
    )
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Disable automatic encoding fallback attempts for CSV files",
    )
    parser.add_argument(
        "--disable-auto-id",
        action="store_true",
        help="Disable automatic identifier derivation when explicit ID columns are unavailable",
    )
    parser.add_argument(
        "--auto-id-prefix-length",
        type=int,
        default=32,
        help="Maximum number of characters to keep from question/answers when deriving automatic IDs (set 0 for unlimited)",
    )
    parser.add_argument(
        "--gt-label-map",
        action="append",
        default=[],
        metavar="SRC=TGT",
        help="Map ground-truth labels to canonical values (can be provided multiple times)",
    )
    parser.add_argument(
        "--pred-label-map",
        action="append",
        default=[],
        metavar="SRC=TGT",
        help="Map prediction labels to canonical values (can be provided multiple times)",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to dump the metrics report as JSON",
    )
    parser.add_argument(
        "--combined-pred-field",
        default="predict",
        help="Field name storing prediction payload when using --combined",
    )
    parser.add_argument(
        "--combined-gt-field",
        default="label",
        help="Field name storing ground-truth payload when using --combined",
    )
    parser.add_argument(
        "--combined-pred-key",
        default="overall_winner",
        help="JSON key inside the prediction payload to extract the predicted label",
    )
    parser.add_argument(
        "--combined-gt-key",
        default="overall_winner",
        help="JSON key inside the ground-truth payload to extract the label",
    )
    parser.add_argument(
        "--combined-split",
        default=None,
        help="Comma-separated counts specifying sequential segments when using --combined (e.g., '60,42')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    combined_mode = bool(args.combined)

    if combined_mode:
        if args.gt or args.pred:
            raise ValueError("--gt/--pred should not be used together with --combined")
        if args.gt_id_column or args.pred_id_column:
            raise ValueError("ID column alignment is unsupported when using --combined")
    else:
        if not args.gt or not args.pred:
            raise ValueError("--gt and --pred are required unless --combined is provided")
        if bool(args.gt_id_column) != bool(args.pred_id_column):
            raise ValueError(
                "Both --gt-id-column and --pred-id-column must be provided together to align by identifiers."
            )

    if args.auto_id_prefix_length < 0:
        raise ValueError("--auto-id-prefix-length must be non-negative")

    gt_label_transform = _make_label_transform(_parse_label_map(args.gt_label_map))
    pred_label_transform = _make_label_transform(_parse_label_map(args.pred_label_map))

    if combined_mode:
        gt_labels, pred_labels = _load_combined_labels(
            path=args.combined,
            pred_field=args.combined_pred_field,
            pred_key=args.combined_pred_key,
            gt_field=args.combined_gt_field,
            gt_key=args.combined_gt_key,
            encoding=args.combined_encoding,
            pred_transform=pred_label_transform,
            gt_transform=gt_label_transform,
        )

        total_rows = len(gt_labels)
        split_counts: List[int] = []
        if args.combined_split:
            try:
                split_counts = [int(part.strip()) for part in args.combined_split.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError("--combined-split must be a comma-separated list of integers") from exc
            if any(count <= 0 for count in split_counts):
                raise ValueError("All values in --combined-split must be positive integers")

        segments: List[Tuple[int, int, str]] = []
        if split_counts:
            start_idx = 0
            for idx, count in enumerate(split_counts, start=1):
                end_idx = start_idx + count
                if end_idx > total_rows:
                    raise ValueError(
                        "Sum of --combined-split counts exceeds number of records in combined file"
                    )
                segments.append((start_idx, end_idx, f"segment_{idx}"))
                start_idx = end_idx
            if start_idx < total_rows:
                segments.append((start_idx, total_rows, f"segment_{len(segments) + 1}"))
        else:
            segments.append((0, total_rows, "segment_1"))

        overall_metrics = _compute_metrics(gt_labels, pred_labels)
        print("=" * 82)
        print(f"Combined file: {args.combined}")
        _print_report(overall_metrics)

        segment_results: List[Dict[str, Any]] = []
        if len(segments) > 1 or split_counts:
            for start_idx, end_idx, name in segments:
                seg_gt = gt_labels[start_idx:end_idx]
                seg_pred = pred_labels[start_idx:end_idx]
                seg_metrics = _compute_metrics(seg_gt, seg_pred)
                print("=" * 82)
                print(
                    f"Segment {name}: rows {start_idx} to {end_idx - 1} (count={end_idx - start_idx})"
                )
                _print_report(seg_metrics)
                segment_results.append(
                    {
                        "name": name,
                        "start": start_idx,
                        "end": end_idx,
                        "count": end_idx - start_idx,
                        "metrics": seg_metrics,
                    }
                )

        report = {
            "combined": args.combined,
            "overall": overall_metrics,
            "segments": segment_results,
        }
    else:
        enable_fallback = not args.disable_fallback
        auto_id_fn: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
        use_ids = bool(args.gt_id_column)

        if not use_ids and not args.disable_auto_id:
            auto_id_fn = _make_auto_id_fn(args.auto_id_prefix_length)
            use_ids = True

        gt_records = _load_labels(
            path=args.gt,
            label_column=args.gt_column,
            id_column=args.gt_id_column,
            encoding=args.gt_encoding,
            enable_fallback=enable_fallback,
            derive_id_fn=auto_id_fn if not args.gt_id_column else None,
            label_transform=gt_label_transform,
        )
        report = {
            "gt": args.gt,
            "predictions": {},
        }

        expanded_pred_paths: List[str] = []
        for path in args.pred:
            if os.path.isdir(path):
                jsonl_files = sorted(
                    glob.glob(os.path.join(path, "*.jsonl")),
                    key=os.path.basename,
                )
                if not jsonl_files:
                    warnings.warn(
                        f"No JSONL files found in directory '{path}'. Skipping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                expanded_pred_paths.extend(jsonl_files)
            else:
                expanded_pred_paths.append(path)

        if not expanded_pred_paths:
            raise RuntimeError("No prediction files provided or discovered after expanding directories.")

        for pred_path in expanded_pred_paths:
            pred_records = _load_labels(
                path=pred_path,
                label_column=args.pred_column,
                id_column=args.pred_id_column,
                encoding=args.pred_encoding,
                enable_fallback=enable_fallback,
                fallback_json_sources=[
                    ("llm_response", "overall_winner"),
                    ("llm_response_json", "overall_winner"),
                    ("response_json", "overall_winner"),
                ],
                derive_id_fn=auto_id_fn if not args.pred_id_column else None,
                label_transform=pred_label_transform,
            )

            gt_labels, pred_labels = _align_labels(
                gt_records,
                pred_records,
                use_ids=use_ids,
            )

            metrics = _compute_metrics(gt_labels, pred_labels)

            print("=" * 82)
            print(f"Prediction file: {pred_path}")
            _print_report(metrics)
            report["predictions"][pred_path] = metrics

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\nMetrics written to {args.report_json}")


if __name__ == "__main__":
    main()


# script
'''
python /path/to/data/LogicJudge/judge_analyze/compute_metrics_ours.py \
  --combined /path/to/data/SFT/LLaMA-Factory-upload/saves/predict/qwen3-30b/full/sft/generated_predictions.jsonl \
  --combined-pred-field predict \
  --combined-gt-field label \
  --combined-split 60,42 \
  --pred-label-map assistant_1=A \
  --pred-label-map assistant_2=B \
  --combined-pred-key overall_winner \
  --combined-gt-key overall_winner \
  --report-json /path/to/data/LogicJudge/judge_analyze/results/metrics_ours_1w.json
python /path/to/data/LogicJudge/judge_analyze/compute_metrics_ours.py \
  --combined /path/to/data/SFT/LLaMA-Factory-upload/saves/predict/qwen3-30b/full/sft_aspect_wtquora/generated_predictions.jsonl \
  --combined-pred-field predict \
  --combined-gt-field label \
  --combined-split 60,42 \
  --pred-label-map assistant_1=A \
  --pred-label-map assistant_2=B \
  --combined-pred-key overall_winner \
  --combined-gt-key overall_winner \
  --report-json /path/to/data/LogicJudge/judge_analyze/results/metrics_ours_2w.json
python /path/to/data/LogicJudge/judge_analyze/compute_metrics_ours.py \
  --combined /path/to/data/SFT/LLaMA-Factory-upload/saves/predict/qwen3-30b/full/sft_base/generated_predictions.jsonl \
  --combined-pred-field predict \
  --combined-gt-field label \
  --combined-split 60,42 \
  --pred-label-map assistant_1=A \
  --pred-label-map assistant_2=B \
  --combined-pred-key overall_winner \
  --combined-gt-key overall_winner \
  --report-json /path/to/data/LogicJudge/judge_analyze/results/metrics_ours_base.json
'''