import argparse
import csv
import gc
import json
import os
import re
from glob import glob
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import torch
import ray
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from aistudio_checkpoint.aistudio_base_checkpointer import lazy_load_checkpoint
import compute_metrics_ours as metrics_utils

CHECKLIST_COUNT = 8
CHECKLIST_COLUMN_PREFIX = "cklist_"
DEFAULT_GT_COLUMN = "overall_winner"

DEFAULT_LOGIC_PROMPT_TEMPLATE = """
You are a logic analysis expert. Given **QUERY** and two answers (A and B), decide which side demonstrates stronger logic and return a structured JSON result.

---

## Task Objectives
1. **Dimension-by-dimension evaluation**: Compare answers A and B across the eight logical dimensions below.
     - Task alignment & claim clarity
     - Global coherence
     - Local coherence
     - Evidence sufficiency & relevance
     - Warrants & causal reasoning
     - Qualifiers & counterpoints
     - Internal consistency
     - Concept introduction & logical transition
2. **Overall assessment**: Combine the dimensions to identify the overall winner and explain why.

---

## Analysis Guidelines
- Focus on logical structure, argumentative strength, and consistency; ignore factual accuracy and writing style.
- Explanations must highlight concrete logical strengths or weaknesses without quoting the original text.
- If both answers excel or both underperform, return "both good" or "both bad" and justify the choice.

---

## Input Format
[INPUT]
- QUERY  
<<<QUERY>>>

- TEXT A  
<<<TEXT_A>>>

- TEXT B  
<<<TEXT_B>>>

---

## Output Format (strict)

Produce the final reply exactly in this format:

```
<think>
{
    "overall_explanation": "A concise English paragraph summarizing the overall logical preference and key reasons (structure, reasoning flow, evidence alignment, balance of qualifiers, concept transitions, etc.). Do not use bullet points."
}
</think>
A>B
```

Requirements:

1. The JSON inside `<think> ... </think>` must contain only the key `"overall_explanation"`.
2. `"overall_explanation"` must be a paragraph in English summarizing the logical evaluation; lists or bullet points are forbidden.
3. After `</think>`, output exactly one final verdict chosen from `"A>B"`, `"B>A"`, `"both good"`, or `"both bad"`.
4. Do not add any extra text, explanations, or punctuation beyond the illustrated format.
""".strip()

LOGIC_PROMPT_TEMPLATE = DEFAULT_LOGIC_PROMPT_TEMPLATE
CSV_CHECKLIST_PREFIX = CHECKLIST_COLUMN_PREFIX
CSV_CHECKLIST_COUNT = CHECKLIST_COUNT
CSV_DEFAULT_LABEL_COLUMN = DEFAULT_GT_COLUMN

CSV_ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "gb18030", "gbk"]

VALUE_PAIR_CANDIDATES = [
    ("answer_a", "answer_b"),
    ("text_a", "text_b"),
    ("response_a", "response_b"),
    ("assistant_1", "assistant_2"),
    ("assistant_a", "assistant_b"),
    ("candidate_a", "candidate_b"),
]

MAX_MODEL_LEN = 32768
MAX_GENERATION_TOKENS = 5000

WINNER_NORMALIZATION_MAP = {
    "assistant_1": "A",
    "assistant 1": "A",
    "a": "A",
    "a>b": "A",
    "a > b": "A",
    "a_win": "A",
    "a wins": "A",
    "assistant_2": "B",
    "assistant 2": "B",
    "b": "B",
    "b>a": "B",
    "b > a": "B",
    "a<b": "B",
    "a < b": "B",
    "b_win": "B",
    "b wins": "B",
    "both good": "both_good",
    "both_good": "both_good",
    "both bad": "both_bad",
    "both_bad": "both_bad",
}


def _safe_load_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = re.sub(r"'", '"', text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return text


def _normalize_winner(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return WINNER_NORMALIZATION_MAP.get(text, text.upper() if text in {"a", "b"} else text)


def _canonicalize_label(value: Any) -> str:
    normalized = _normalize_winner(value)
    if normalized in {"a", "b"}:
        return normalized.upper()
    if normalized in {"A", "B"}:
        return normalized
    if normalized in {"both_good", "both_bad"}:
        return normalized
    if normalized.lower() in {"a", "assistant_1"}:
        return "A"
    if normalized.lower() in {"b", "assistant_2"}:
        return "B"
    return normalized


def _raw_from_canonical(label: str) -> str:
    if label == "A":
        return "A>B"
    if label == "B":
        return "B>A"
    if label == "both_good":
        return "both good"
    if label == "both_bad":
        return "both bad"
    return label


_SAFE_LOAD_JSON = _safe_load_json
_CANONICALIZE_LABEL = _canonicalize_label
_RAW_FROM_CANONICAL = _raw_from_canonical


def _select_first_available(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


_SELECT_FIRST_AVAILABLE = _select_first_available


def _invert_canonical_label(label: str) -> str:
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    return label


def _infer_swap_pairs(fieldnames: Sequence[str]) -> List[Tuple[str, str]]:
    field_set = {name for name in fieldnames if isinstance(name, str)}
    pairs: List[Tuple[str, str]] = []

    for left, right in VALUE_PAIR_CANDIDATES:
        if left in field_set and right in field_set:
            pairs.append((left, right))

    suffix_patterns = [
        ("_a", "_b"),
        ("_A", "_B"),
        ("-a", "-b"),
        ("-A", "-B"),
        ("_1", "_2"),
        ("1", "2"),
    ]

    seen: Set[Tuple[str, str]] = set()
    for name in field_set:
        for suffix_a, suffix_b in suffix_patterns:
            if not name.endswith(suffix_a):
                continue
            base = name[: -len(suffix_a)]
            candidate = f"{base}{suffix_b}"
            if candidate in field_set:
                key = tuple(sorted((name, candidate)))
                if key not in seen:
                    seen.add(key)
                    pairs.append((name, candidate))
    return pairs


def _swap_row_entries(row: Dict[str, Any], pairs: Sequence[Tuple[str, str]]) -> None:
    seen: Set[Tuple[str, str]] = set()
    for left, right in pairs:
        key = (left, right) if left <= right else (right, left)
        if key in seen:
            continue
        seen.add(key)
        if left in row and right in row:
            row[left], row[right] = row[right], row[left]


def _flip_preference_string(value: Any) -> str:
    canonical = _CANONICALIZE_LABEL(value)
    if canonical == "A":
        return _RAW_FROM_CANONICAL("B")
    if canonical == "B":
        return _RAW_FROM_CANONICAL("A")
    if canonical in {"both_good", "both_bad"}:
        return _RAW_FROM_CANONICAL(canonical)

    text = str(value)
    swapped = (
        text.replace("A>B", "__TMP__")
        .replace("B>A", "A>B")
        .replace("__TMP__", "B>A")
        .replace("assistant_1", "__ASSIST__")
        .replace("assistant_2", "assistant_1")
        .replace("__ASSIST__", "assistant_2")
    )
    if swapped != text:
        return swapped

    lowered = text.strip().lower()
    if lowered == "a":
        return "B"
    if lowered == "b":
        return "A"
    return text


def _flip_checklist_labels(row: Dict[str, Any], additional_keys: Optional[Sequence[str]] = None) -> None:
    keys: List[str] = [f"{CSV_CHECKLIST_PREFIX}{idx}" for idx in range(1, CSV_CHECKLIST_COUNT + 1)]
    if additional_keys:
        keys.extend(additional_keys)

    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if value in (None, ""):
            continue
        decoded = _SAFE_LOAD_JSON(value)
        if isinstance(decoded, dict):
            preference_value = decoded.get("winner")
            canonical = _CANONICALIZE_LABEL(preference_value)
            if canonical == "A":
                decoded["winner"] = _RAW_FROM_CANONICAL("B")
            elif canonical == "B":
                decoded["winner"] = _RAW_FROM_CANONICAL("A")
            elif canonical in {"both_good", "both_bad"}:
                decoded["winner"] = _RAW_FROM_CANONICAL(canonical)
            if "winner" in decoded:
                decoded.pop("winner")
            row[key] = json.dumps(decoded, ensure_ascii=False)
        else:
            row[key] = _flip_preference_string(decoded)


_INVERT_CANONICAL_LABEL = _invert_canonical_label
_INFER_SWAP_PAIRS = _infer_swap_pairs
_SWAP_ROW_ENTRIES = _swap_row_entries
_FLIP_CHECKLIST_LABELS = _flip_checklist_labels


def _load_csv_rows(path: str, max_rows: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    last_error: Optional[Exception] = None
    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            with open(path, "r", encoding=encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    rows.append(dict(row))
                    if max_rows is not None and len(rows) >= max_rows:
                        break
            if rows or max_rows == 0:
                break
        except UnicodeDecodeError as exc:
            last_error = exc
            rows = []
            continue
    else:
        raise RuntimeError(
            f"Unable to decode CSV {path} with encodings {CSV_ENCODING_CANDIDATES}"
        ) from last_error

    if max_rows is not None and len(rows) > max_rows:
        rows = rows[:max_rows]

    if rows and "sample_id" not in rows[0]:
        for idx, row in enumerate(rows):
            row.setdefault("sample_id", str(idx))
        if "sample_id" not in fieldnames:
            fieldnames = list(fieldnames) + ["sample_id"]
    return rows, fieldnames


def _build_prompt_from_row(
    row: Mapping[str, Any], template: str
) -> Tuple[str, List[Any], str, str, str]:
    prompt_template = template or LOGIC_PROMPT_TEMPLATE
    query = _SELECT_FIRST_AVAILABLE(row, ["query", "question", "prompt", "QUERY"])
    answer_a = _SELECT_FIRST_AVAILABLE(
        row,
        ["answer_a", "text_a", "response_a", "assistant_1", "assistant_a"],
    )
    answer_b = _SELECT_FIRST_AVAILABLE(
        row,
        ["answer_b", "text_b", "response_b", "assistant_2", "assistant_b"],
    )

    checklist_items: List[Any] = []
    # if "<<<CHECKLIST_JSON>>>" in prompt_template:
    #     for idx in range(1, CSV_CHECKLIST_COUNT + 1):
    #         key = f"{CSV_CHECKLIST_PREFIX}{idx}"
    #         raw_value = row.get(key)
    #         if raw_value in (None, ""):
    #             continue
    #         checklist_items.append(_SAFE_LOAD_JSON(raw_value))
    #     checklist_payload = json.dumps(checklist_items, ensure_ascii=False, indent=2)
    # else:
    #     checklist_payload = ""

    prompt = prompt_template.replace("<<<QUERY>>>", query)
    prompt = prompt.replace("<<<TEXT_A>>>", answer_a)
    prompt = prompt.replace("<<<TEXT_B>>>", answer_b)
    # if "<<<CHECKLIST_JSON>>>" in prompt:
    #     prompt = prompt.replace("<<<CHECKLIST_JSON>>>", checklist_payload)

    return prompt.strip(), checklist_items, query, answer_a, answer_b


def _load_csv_dataset(
    path: str,
    label_column: str,
    input_key: str,
    prompt_template: Optional[str],
    max_rows: Optional[int],
) -> List[Dict[str, Any]]:
    template = (prompt_template or LOGIC_PROMPT_TEMPLATE).strip()
    rows, fieldnames = _load_csv_rows(path, max_rows=max_rows)
    records: List[Dict[str, Any]] = []

    # Treat rows as already paired when they include explicit variant markers (e.g., generated by generate_winner).
    data_is_precombined = any(
        ("::" in str(row.get("sample_id", "")))
        or (str(row.get("variant", "")).lower() in {"original", "swapped"})
        for row in rows
    )

    swap_pairs = _INFER_SWAP_PAIRS(fieldnames)
    if ("answer_a", "answer_b") not in swap_pairs and ("answer_b", "answer_a") not in swap_pairs:
        if "answer_a" in fieldnames and "answer_b" in fieldnames:
            swap_pairs = list(swap_pairs) + [("answer_a", "answer_b")]

    preference_columns = [
        name for name in fieldnames if isinstance(name, str) and "which answer better satisfies" in name
    ]

    def _resolve_label_value(canonical_label: str, original_value: Any) -> str:
        if canonical_label:
            return _RAW_FROM_CANONICAL(canonical_label)
        return str(original_value or "").strip()

    def _finalize_record(
        row_data: Dict[str, Any],
        *,
        sample_id: str,
        base_sample_id: str,
        variant: str,
        label_value: str,
    ) -> Dict[str, Any]:
        working_row = dict(row_data)
        working_row[label_column] = label_value
        prompt_text, checklist_items, query, answer_a, answer_b = _build_prompt_from_row(working_row, template)
        record = dict(working_row)
        record[input_key] = prompt_text
        record["sample_id"] = sample_id
        record["base_sample_id"] = base_sample_id
        record["variant"] = variant
        record["query"] = query
        record["answer_a"] = answer_a
        record["answer_b"] = answer_b
        record["checklist"] = checklist_items
        record["label"] = label_value
        record["overall_winner"] = label_value
        record["ground_truth"] = label_value
        record["prompt_template"] = template
        record["source_format"] = "csv"
        return record

    for idx, row in enumerate(rows):
        base_sample_id = str(row.get("base_sample_id") or row.get("sample_id") or idx)
        canonical_label = _CANONICALIZE_LABEL(row.get(label_column))
        label_value = _resolve_label_value(canonical_label, row.get(label_column))

        if data_is_precombined:
            sample_id = str(row.get("sample_id") or f"{base_sample_id}::original")
            variant = str(row.get("variant") or ("swapped" if sample_id.endswith("::swapped") else "original"))
            records.append(
                _finalize_record(
                    row,
                    sample_id=sample_id,
                    base_sample_id=base_sample_id,
                    variant=variant,
                    label_value=label_value,
                )
            )
            continue

        # Ensure downstream metrics see paired samples with deterministic identifiers.
        original_sample_id = f"{base_sample_id}::original"
        records.append(
            _finalize_record(
                row,
                sample_id=original_sample_id,
                base_sample_id=base_sample_id,
                variant="original",
                label_value=label_value,
            )
        )

        swapped_row = dict(row)
        _SWAP_ROW_ENTRIES(swapped_row, swap_pairs)
        _FLIP_CHECKLIST_LABELS(swapped_row, preference_columns)
        swapped_canonical = _INVERT_CANONICAL_LABEL(canonical_label)
        swapped_label_value = _resolve_label_value(swapped_canonical, swapped_row.get(label_column))
        swapped_sample_id = f"{base_sample_id}::swapped"
        records.append(
            _finalize_record(
                swapped_row,
                sample_id=swapped_sample_id,
                base_sample_id=base_sample_id,
                variant="swapped",
                label_value=swapped_label_value,
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference over JSONL benchmarks using vLLM.")
    parser.add_argument(
        "--input-files",
        required=True,
        help="Comma-separated list of JSONL/CSV files or glob patterns to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write evaluation outputs.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local or aistudio path to the model checkpoint.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size (H200=1, H800/A100=2).",
    )
    parser.add_argument(
        "--input-key",
        default="input",
        help="Key in each JSON record to use as the prompt (default: input).",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Optional Ray address; leave empty to start a local cluster.",
    )
    parser.add_argument(
        "--report-performance",
        action="store_true",
        help="Compute agreement/precision/recall/F1 for choices and save a report alongside outputs.",
    )
    parser.add_argument(
        "--csv-input",
        default=None,
        help=(
            "Optional CSV file (same format as generate_winner ground truth) whose rows will be merged into the inputs. "
            "Prompts are built from query/answer_a/answer_b and included in performance reporting."
        ),
    )
    parser.add_argument(
        "--csv-label-column",
        default=CSV_DEFAULT_LABEL_COLUMN,
        help="Label column name in the CSV when --csv-input is provided (default mirrors generate_winner).",
    )
    parser.add_argument(
        "--csv-prompt-template",
        default=None,
        help="Override prompt template for CSV rows. Use placeholders <<<QUERY>>>, <<<TEXT_A>>>, <<<TEXT_B>>>.",
    )
    parser.add_argument(
        "--csv-max-rows",
        type=int,
        default=None,
        help="Optional limit on number of CSV rows processed (useful for quick smoke tests).",
    )
    return parser.parse_args()


def _expand_input_files(raw_patterns: str) -> list[str]:
    patterns = [chunk.strip() for chunk in raw_patterns.split(",") if chunk.strip()]
    resolved: list[str] = []
    missing: list[str] = []
    for pattern in patterns:
        matches = sorted(glob(pattern))
        if matches:
            resolved.extend(matches)
        elif os.path.exists(pattern):
            resolved.append(pattern)
        else:
            missing.append(pattern)
    if not resolved:
        raise ValueError(f"No input files found for patterns: {raw_patterns}")
    if missing:
        print(f"Warning: the following patterns produced no matches and will be ignored: {missing}")
    return resolved


def get_local_path(model_path: str) -> str:
    if not model_path.startswith("aistudio:"):
        return model_path
    return lazy_load_checkpoint(resuming_path=model_path)


def read_jsonl(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
def write_jsonl(objs, data_path):
    with open(data_path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


CHOICE_CANONICALS = ("A>B", "A<B", "both good", "both bad")
CHOICE_PATTERNS = [
    (canonical, re.compile(re.escape(canonical), flags=re.IGNORECASE))
    for canonical in CHOICE_CANONICALS
]
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def _canonicalize_choice(value: str) -> str:
    normalized = metrics_utils._normalize_label(value)
    if not normalized:
        return ""

    for canonical, pattern in CHOICE_PATTERNS:
        if pattern.search(normalized):
            return canonical

    lowered = normalized.lower()
    if re.search(r"a\s*>\s*b", lowered):
        return "A>B"
    if re.search(r"a\s*<\s*b", lowered):
        return "A<B"
    if "both" in lowered and "good" in lowered:
        return "both good"
    if "both" in lowered and "bad" in lowered:
        return "both bad"
    if "assistant" in lowered:
        if "1" in lowered or "a" in lowered:
            return "A>B"
        if "2" in lowered or "b" in lowered:
            return "A<B"
    if lowered in {"a", "option a", "choice a"}:
        return "A>B"
    if lowered in {"b", "option b", "choice b"}:
        return "A<B"
    return normalized


def _extract_choice_from_text(text: str) -> str:
    if not text:
        return ""
    cleaned = THINK_BLOCK_PATTERN.sub(" ", text)
    matches: list[tuple[int, str]] = []
    for canonical, pattern in CHOICE_PATTERNS:
        for match in pattern.finditer(cleaned):
            matches.append((match.start(), canonical))
    if not matches:
        return _canonicalize_choice(text)
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def _extract_label_choice(record: dict) -> str:
    raw = record.get("label")
    if isinstance(raw, str):
        return _canonicalize_choice(raw)
    if raw is not None:
        extracted = metrics_utils._extract_label_from_payload(raw, "overall_winner")
        if extracted:
            return _canonicalize_choice(extracted)
    return ""


def _extract_prediction_choice(record: dict) -> str:
    response = record.get("response")
    if response is None and isinstance(record.get("judge_results"), list):
        try:
            response = record["judge_results"][-1]["response"]
        except (IndexError, KeyError, TypeError):
            response = None
    if isinstance(response, str):
        return _extract_choice_from_text(response)
    if response is not None:
        try:
            response_text = json.dumps(response, ensure_ascii=False)
        except TypeError:
            response_text = str(response)
        if response_text:
            return _extract_choice_from_text(response_text)
    return ""


def _compute_dataset_metrics(records: list[dict]) -> tuple[dict, list[str], list[str]]:
    truth_by_sample: Dict[str, str] = {}
    truth_by_base: Dict[str, Dict[str, str]] = {}
    predictions_by_sample: Dict[str, str] = {}
    predictions_by_base: Dict[str, Dict[str, str]] = {}

    missing_gt = 0
    missing_pred = 0

    for idx, record in enumerate(records):
        sample_id = str(record.get("sample_id") or idx)
        base_sample_id = str(
            record.get("base_sample_id")
            or record.get("sample_base_id")
            or sample_id.split("::")[0]
        )
        variant = str(
            record.get("variant")
            or ("swapped" if sample_id.endswith("::swapped") else "original")
        )

        gt_choice = _extract_label_choice(record)
        pred_choice = _extract_prediction_choice(record)

        canonical_gt = _CANONICALIZE_LABEL(gt_choice)
        canonical_pred = _CANONICALIZE_LABEL(pred_choice)

        if not canonical_gt:
            missing_gt += 1
        if not canonical_pred:
            missing_pred += 1

        truth_by_sample[sample_id] = canonical_gt
        truth_by_base.setdefault(base_sample_id, {})[variant] = canonical_gt

        if canonical_pred:
            predictions_by_sample[sample_id] = canonical_pred
            predictions_by_base.setdefault(base_sample_id, {})[variant] = canonical_pred

    expected_sample_ids = list(truth_by_sample.keys())
    missing_sample_ids = [sid for sid in expected_sample_ids if sid not in predictions_by_sample]

    aligned_gt: List[str] = []
    aligned_pred: List[str] = []
    for sample_id in expected_sample_ids:
        pred_label = predictions_by_sample.get(sample_id)
        if pred_label in (None, ""):
            continue
        aligned_gt.append(truth_by_sample[sample_id])
        aligned_pred.append(pred_label)

    if aligned_gt:
        variant_metrics = metrics_utils._compute_metrics(aligned_gt, aligned_pred)
    else:
        variant_metrics = {
            "total": 0,
            "labels": {},
            "agreement": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
        }

    pair_total = len(truth_by_base)
    pair_correct = 0
    pairs_with_missing_variants = 0
    variant_missing_count = 0
    pairs_with_all_variants_predicted = 0
    pair_prediction_mismatch_count = 0
    pair_prediction_mismatch_ids: List[str] = []

    for base_sample_id, truth_variants in truth_by_base.items():
        predicted_variants = predictions_by_base.get(base_sample_id, {})
        all_variants_present = True
        all_variants_correct = True
        collected_predictions: Dict[str, str] = {}
        for variant_name, truth_label in truth_variants.items():
            pred_label = predicted_variants.get(variant_name)
            if pred_label in (None, ""):
                all_variants_present = False
                all_variants_correct = False
                variant_missing_count += 1
                continue
            collected_predictions[variant_name] = pred_label
            if pred_label != truth_label:
                all_variants_correct = False
        if not all_variants_present:
            pairs_with_missing_variants += 1
        if all_variants_present and all_variants_correct:
            pair_correct += 1
        if all_variants_present:
            pairs_with_all_variants_predicted += 1
            unique_predictions = set(collected_predictions.values())
            if len(unique_predictions) > 1:
                pair_prediction_mismatch_count += 1
                pair_prediction_mismatch_ids.append(base_sample_id)

    pair_accuracy = pair_correct / pair_total if pair_total else 0.0

    metrics: Dict[str, Any] = dict(variant_metrics)
    metrics["agreement"] = pair_accuracy
    metrics["pair_accuracy"] = pair_accuracy
    metrics["pair_correct"] = pair_correct
    metrics["pair_total"] = pair_total
    metrics["pairs_with_missing_variants"] = pairs_with_missing_variants
    metrics["variant_missing_count"] = variant_missing_count
    metrics["pairs_with_all_variants_predicted"] = pairs_with_all_variants_predicted
    metrics["pair_prediction_mismatch_count"] = pair_prediction_mismatch_count
    metrics["pair_prediction_mismatch_rate"] = (
        pair_prediction_mismatch_count / pairs_with_all_variants_predicted
        if pairs_with_all_variants_predicted
        else 0.0
    )
    metrics["pair_prediction_mismatch_ids"] = pair_prediction_mismatch_ids[:50]
    metrics["evaluated_samples"] = len(aligned_gt)
    metrics["expected_samples"] = len(expected_sample_ids)
    metrics["missing_samples"] = len(missing_sample_ids)
    metrics["missing_sample_ids"] = missing_sample_ids[:50]
    metrics["missing_ground_truth"] = missing_gt
    metrics["missing_prediction"] = missing_pred
    metrics["variant_level"] = variant_metrics

    return metrics, aligned_gt, aligned_pred


@ray.remote
class InferEngine:
    def __init__(self, model_path: str, tp: int, engine_idx: int):
        self.llm = LLM(
            model=get_local_path(model_path),
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.9,
            max_model_len=MAX_MODEL_LEN,
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=1,
            top_k=-1,
            max_tokens=MAX_GENERATION_TOKENS,
        )
        self.engine_idx = engine_idx
        
    def generate(self, prompts: list[str]) -> tuple[list[str], list[int], list[int]]:
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=self.engine_idx == 0)
        response_texts: list[str] = []
        response_lengths: list[int] = []
        prompt_token_lengths: list[int] = []
        for output in outputs:
            prompt_token_ids = getattr(output, "prompt_token_ids", None) or []
            prompt_token_lengths.append(len(prompt_token_ids))
            if output.outputs:
                first_output = output.outputs[0]
                response_texts.append(first_output.text)
                response_lengths.append(len(getattr(first_output, "token_ids", []) or []))
            else:
                response_texts.append("")
                response_lengths.append(0)
        return response_texts, response_lengths, prompt_token_lengths

    def close(self):
        # Release the underlying vLLM model to free GPU memory
        del self.llm
        self.llm = None
        gc.collect()
        torch.cuda.empty_cache()

def main():
    args = parse_args()

    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    input_files = _expand_input_files(args.input_files)
    model_path = args.model_path
    tp = args.tp
    input_key = args.input_key

    n_gpus = int(ray.cluster_resources().get("GPU", 0))
    assert (n_gpus >= tp) and (n_gpus % tp == 0), "The cluster must expose a GPU count that is a multiple of tp"
    n_engines = n_gpus // tp

    tokenizer = AutoTokenizer.from_pretrained(get_local_path(model_path))
    engines = [InferEngine.options(num_gpus=tp).remote(model_path, tp, i) for i in range(n_engines)]
    print(f"Spinning up {n_engines} inference engines with {tp} GPU(s) each")

    start = 0
    prompts: List[str] = []
    total_data_list: List[Dict[str, Any]] = []
    indexes: Dict[str, Tuple[int, int]] = {}
    requested_prompt_token_lengths: List[int] = []

    def _dedupe_dataset_name(base: str) -> str:
        candidate = base
        suffix = 2
        while candidate in indexes:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _append_dataset(records: List[Dict[str, Any]], dataset_name: str, origin: str) -> None:
        nonlocal start
        if not records:
            print(f"Warning: {origin} produced no samples; skipping dataset.")
            return
        for record in records:
            user_content = record.get(input_key, "")
            if not isinstance(user_content, str):
                user_content = "" if user_content is None else str(user_content)
            messages = [{"role": "user", "content": user_content}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            token_count = len(prompt_token_ids)
            prompts.append(prompt_text)
            requested_prompt_token_lengths.append(token_count)
            is_overflow = token_count > MAX_MODEL_LEN
            record["input_truncated"] = is_overflow
            record["prompt_token_count_requested"] = token_count
            record["prompt_token_count_used"] = None if is_overflow else token_count
        total_data_list.extend(records)
        end = start + len(records)
        indexes[dataset_name] = (start, end)
        start = end
        print(f"File: {origin} yielded {len(records)} samples")

    for input_file_path in input_files:
        lower_path = input_file_path.lower()
        dataset_base = os.path.splitext(os.path.basename(input_file_path))[0]
        dataset_name = _dedupe_dataset_name(dataset_base)
        if lower_path.endswith(".csv"):
            csv_records = _load_csv_dataset(
                path=input_file_path,
                label_column=args.csv_label_column,
                input_key=input_key,
                prompt_template=args.csv_prompt_template,
                max_rows=args.csv_max_rows,
            )
            _append_dataset(csv_records, dataset_name, f"{input_file_path} (csv)")
        else:
            data_list = read_jsonl(input_file_path)
            _append_dataset(data_list, dataset_name, input_file_path)

    if args.csv_input and args.csv_input not in input_files:
        extra_dataset_name = _dedupe_dataset_name(
            os.path.splitext(os.path.basename(args.csv_input))[0]
        )
        extra_records = _load_csv_dataset(
            path=args.csv_input,
            label_column=args.csv_label_column,
            input_key=input_key,
            prompt_template=args.csv_prompt_template,
            max_rows=args.csv_max_rows,
        )
        _append_dataset(extra_records, extra_dataset_name, f"{args.csv_input} (csv-input)")

    if not prompts:
        raise ValueError("No samples were loaded. Please verify --input-files / --csv-input paths.")

    print(f"Loaded {len(prompts)} samples in total")

    # Split prompts evenly across engines; distribute remainders by engine index
    chunks = [
        [p for j, p in enumerate(prompts) if j % n_engines == i]
        for i in range(n_engines)
    ]

    result_refs = [engines[i].generate.remote(chunk) for i, chunk in enumerate(chunks)]
    chunk_results = ray.get(result_refs)

    # Merge inference results and persist them
    for i in range(n_engines):
        rsp_texts, rsp_lengths, prompt_used_lengths = chunk_results[i]
        for j in range(len(rsp_texts)):
            idx = j * n_engines + i
            total_data_list[idx]['response'] = rsp_texts[j]
            total_data_list[idx]['response_length'] = rsp_lengths[j]
            total_data_list[idx]['judge_model'] = model_path

            requested_len = (
                requested_prompt_token_lengths[idx]
                if idx < len(requested_prompt_token_lengths)
                else None
            )
            actual_len = (
                prompt_used_lengths[j] if j < len(prompt_used_lengths) else None
            )
            used_len = actual_len if actual_len is not None else requested_len
            if used_len is not None:
                total_data_list[idx]['prompt_token_count_used'] = used_len
            if (
                actual_len is not None
                and requested_len is not None
                and actual_len < requested_len
            ):
                total_data_list[idx]['input_truncated'] = True

            total_data_list[idx]['model_answer'] = _extract_prediction_choice(total_data_list[idx])

    # write_jsonl(data_list, output_file_path)
    model_name = os.path.basename(model_path)
    performance_report = None
    overall_gt: list[str] = []
    overall_pred: list[str] = []
    overall_missing_gt = 0
    overall_missing_pred = 0
    overall_pair_correct = 0
    overall_pair_total = 0
    overall_pairs_with_missing = 0
    overall_variant_missing = 0
    overall_evaluated_samples = 0
    overall_expected_samples = 0
    overall_missing_sample_ids: list[str] = []
    overall_pairs_with_all_predictions = 0
    overall_prediction_mismatches = 0
    overall_prediction_mismatch_ids: list[str] = []
    if args.report_performance:
        performance_report = {
            "model_path": model_path,
            "model_name": model_name,
            "datasets": {},
        }
    os.makedirs(args.output_dir, exist_ok=True)
    for testset, (start, end) in indexes.items():
        output_file_path = os.path.join(args.output_dir, f"{testset}_{model_name}.jsonl")
        dataset_records = total_data_list[start: end]
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            json.dump(dataset_records, fout, ensure_ascii=False, indent=2)
        if performance_report is not None:
            metrics, gt_choices, pred_choices = _compute_dataset_metrics(dataset_records)
            performance_report["datasets"][testset] = {
                "count": len(dataset_records),
                "metrics": metrics,
            }
            overall_gt.extend(gt_choices)
            overall_pred.extend(pred_choices)
            overall_missing_gt += metrics.get("missing_ground_truth", 0)
            overall_missing_pred += metrics.get("missing_prediction", 0)
            overall_pair_correct += metrics.get("pair_correct", 0)
            overall_pair_total += metrics.get("pair_total", 0)
            overall_pairs_with_missing += metrics.get("pairs_with_missing_variants", 0)
            overall_variant_missing += metrics.get("variant_missing_count", 0)
            overall_pairs_with_all_predictions += metrics.get("pairs_with_all_variants_predicted", 0)
            overall_prediction_mismatches += metrics.get("pair_prediction_mismatch_count", 0)
            overall_evaluated_samples += metrics.get("evaluated_samples", 0)
            overall_expected_samples += metrics.get("expected_samples", 0)
            overall_missing_sample_ids.extend(metrics.get("missing_sample_ids", []))
            overall_prediction_mismatch_ids.extend(metrics.get("pair_prediction_mismatch_ids", []))

    if performance_report is not None and overall_expected_samples:
        if overall_gt:
            variant_metrics = metrics_utils._compute_metrics(overall_gt, overall_pred)
        else:
            variant_metrics = {
                "total": 0,
                "labels": {},
                "agreement": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
            }

        pair_accuracy = overall_pair_correct / overall_pair_total if overall_pair_total else 0.0
        overall_metrics: Dict[str, Any] = dict(variant_metrics)
        overall_metrics["agreement"] = pair_accuracy
        overall_metrics["pair_accuracy"] = pair_accuracy
        overall_metrics["pair_correct"] = overall_pair_correct
        overall_metrics["pair_total"] = overall_pair_total
        overall_metrics["pairs_with_missing_variants"] = overall_pairs_with_missing
        overall_metrics["variant_missing_count"] = overall_variant_missing
        overall_metrics["pairs_with_all_variants_predicted"] = overall_pairs_with_all_predictions
        overall_metrics["pair_prediction_mismatch_count"] = overall_prediction_mismatches
        overall_metrics["pair_prediction_mismatch_rate"] = (
            overall_prediction_mismatches / overall_pairs_with_all_predictions
            if overall_pairs_with_all_predictions
            else 0.0
        )
        overall_metrics["missing_ground_truth"] = overall_missing_gt
        overall_metrics["missing_prediction"] = overall_missing_pred
        overall_metrics["missing_samples"] = len(overall_missing_sample_ids)
        overall_metrics["missing_sample_ids"] = list(dict.fromkeys(overall_missing_sample_ids))[:50]
        overall_metrics["pair_prediction_mismatch_ids"] = list(
            dict.fromkeys(overall_prediction_mismatch_ids)
        )[:50]
        overall_metrics["evaluated_samples"] = overall_evaluated_samples
        overall_metrics["expected_samples"] = overall_expected_samples
        overall_metrics["variant_level"] = variant_metrics

        performance_report["overall"] = {
            "count": overall_expected_samples,
            "metrics": overall_metrics,
        }
        report_path = os.path.join(args.output_dir, f"{model_name}_performance.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(performance_report, fh, ensure_ascii=False, indent=2)
        print(f"Performance metrics saved to {report_path}")
    elif performance_report is not None:
        print("Performance reporting requested, but no samples were processed.")

    # Ask remote engines to release GPU memory first
    ray.get([engine.close.remote() for engine in engines])

    # Then terminate the remote actors
    for engine in engines:
        ray.kill(engine)

    # If this script launched a local Ray cluster, shut it down
    if not args.ray_address:
        ray.shutdown()

    # The driver rarely holds GPU memory, but clearing it is harmless maintenance
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()



#  v4-SFT
'''
python vllm_infer_swap.py \
  --input-files /path/to/data/data/zhihu/GT/zhihu_gt.csv \
  --output-dir /path/to/data/RL/evaluate/zhihu_new_mirror \
  --model-path aistudio://14681891/sft_lasteval_5epoch_v4-step-134-ckpt_hf-step134 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate/deepresearch_new_mirror \
  --model-path aistudio://14681891/sft_lasteval_5epoch_v4-step-134-ckpt_hf-step134 \
  --tp 2 \
  --input-key input \
  --report-performance \
  

python vllm_infer_swap.py \
  --input-files /path/to/data/data/zhihu/GT/zhihu_gt.csv \
  --output-dir /path/to/data/RL/evaluate/zhihu_new_mirror \
  --model-path aistudio://14682121/sft_lasteval_5epoch_v4-step-268-ckpt_hf-step268 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate/deepresearch_new_mirror \
  --model-path aistudio://14682121/sft_lasteval_5epoch_v4-step-268-ckpt_hf-step268 \
  --tp 2 \
  --input-key input \
  --report-performance \
  
'''


#  v6-SFT
'''
python vllm_infer_swap.py \
  --input-files /path/to/data/data/zhihu/GT/zhihu_gt.csv \
  --output-dir /path/to/data/RL/evaluate/zhihu_new_mirror \
  --model-path aistudio://14682492/sft_lasteval_5epoch_v6-step-696-ckpt_hf-step696 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate/deepresearch_new_mirror \
  --model-path aistudio://14682492/sft_lasteval_5epoch_v6-step-696-ckpt_hf-step696 \
  --tp 2 \
  --input-key input \
  --report-performance \
  
python vllm_infer_swap.py \
  --input-files /path/to/data/data/zhihu/GT/zhihu_gt.csv \
  --output-dir /path/to/data/RL/evaluate/zhihu_new_mirror \
  --model-path aistudio://14682822/sft_lasteval_5epoch_v6-step-928-ckpt_hf-step928 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate/deepresearch_new_mirror \
  --model-path aistudio://14682822/sft_lasteval_5epoch_v6-step-928-ckpt_hf-step928 \
  --tp 2 \
  --input-key input \
  --report-performance \

python vllm_infer_swap.py \
  --input-files /path/to/data/data/zhihu/GT/zhihu_gt.csv \
  --output-dir /path/to/data/RL/evaluate/zhihu_new_mirror \
  --model-path aistudio://14683387/sft_lasteval_5epoch_v6-step-1160-ckpt_hf-step1160 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate/deepresearch_new_mirror \
  --model-path aistudio://14683387/sft_lasteval_5epoch_v6-step-1160-ckpt_hf-step1160 \
  --tp 2 \
  --input-key input \
  --report-performance \

python vllm_infer_swap.py \
  --input-files /path/to/data/data/zhihu/GT/zhihu_gt.csv \
  --output-dir /path/to/data/RL/evaluate/zhihu_new_mirror \
  --model-path aistudio://14333845/Qwen3-30B-A3B-gemm_hf-step0 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate/deepresearch_new_mirror \
  --model-path aistudio://10841644//pcache-mnt/rw/checkpoint/54124/416544/192862771/250731144848/1000/ \
  --tp 2 \
  --input-key input \
  --report-performance


python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate/deepresearch_new_mirror \
  --model-path aistudio://14333845/Qwen3-30B-A3B-gemm_hf-step0 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/zhihu/GT/zhihu_gt.csv \
  --output-dir /path/to/data/RL/evaluate/zhihu_new_mirror \
  --model-path aistudio://14682989/sft_lasteval_5epoch_v6-step-464-ckpt_hf-step464 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate/deepresearch_new_mirror \
  --model-path aistudio://14682989/sft_lasteval_5epoch_v6-step-464-ckpt_hf-step464 \
  --tp 2 \
  --input-key input \
  --report-performance \

python vllm_infer_swap.py \
  --input-files /path/to/data/data/zhihu/GT/zhihu_gt.csv \
  --output-dir /path/to/data/RL/evaluate/zhihu_new_mirror \
  --model-path aistudio://14683230/sft_lasteval_5epoch_v6-step-232-ckpt_hf-step232 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate/deepresearch_new_mirror \
  --model-path aistudio://14683230/sft_lasteval_5epoch_v6-step-232-ckpt_hf-step232 \
  --tp 2 \
  --input-key input \
  --report-performance \
'''

#  v5-SFT
'''
python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate/deepresearch_new_mirror \
  --model-path aistudio://14683724/sft_lasteval_5epoch_v5-step-708-ckpt_hf-step708 \
  --tp 2 \
  --input-key input \
  --report-performance


'''

#  v2-RL
'''

python vllm_infer_swap.py \
  --input-files /path/to/data/data/zhihu/GT/zhihu_gt.csv \
  --output-dir /path/to/data/RL/evaluate_backup/zhihu \
  --model-path aistudio://14712798/sft_lasteval_3epoch_v2_wo_len-step-100-ckpt_hf-step100 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate_backup \
  --model-path aistudio://14712798/sft_lasteval_3epoch_v2_wo_len-step-100-ckpt_hf-step100 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/zhihu/GT/zhihu_gt.csv \
  --output-dir /path/to/data/RL/evaluate_backup/zhihu \
  --model-path aistudio://14712662/sft_lasteval_3epoch_v2-step-100-ckpt_hf-step100 \
  --tp 2 \
  --input-key input \
  --report-performance

python vllm_infer_swap.py \
  --input-files /path/to/data/data/deep_research/GT/deep_research_gt.csv \
  --output-dir /path/to/data/RL/evaluate_backup \
  --model-path aistudio://14712662/sft_lasteval_3epoch_v2-step-100-ckpt_hf-step100 \
  --tp 2 \
  --input-key input \
  --report-performance
'''