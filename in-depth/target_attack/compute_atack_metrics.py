#!/usr/bin/env python3
"""Run LLM-as-judge over checklist prompts and compute targeted attack metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from datetime import datetime
import math

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore[assignment]

try:
    import pandas as _pd_debug  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    _pd_debug = None

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - optional dependency
    _tqdm = None


class _SilentTqdm:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        self.total = kwargs.get("total")

    def update(self, _: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


LOGGER = logging.getLogger(__name__)


DEFAULT_API_KEY_ENV = "ANT_OPENAI_API_KEY_V2"
DEFAULT_BASE_URL = os.environ.get("ANT_OPENAI_BASE_URL", "http://matrixllm-pool.global.alipay.com/v1/")
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_REQUEST_TIMEOUT = 180
ASPECT_NAMES = [
    "Task alignment & claim clarity（Task alignment & claim clarity）",
    "Global coherence（Global coherence）",
    "Local coherence（Local coherence）",
    "Evidence sufficiency & relevance（Evidence sufficiency & relevance）",
    "Warrants & causal reasoning（Warrants & causal reasoning）",
    "Qualifiers & counterpoints（Qualifiers & counterpoints）",
    "Internal consistency（Internal consistency）",
    "Concept introduction & logical transition（Concept introduction & logical transition）",
]
ASPECT_KEYS = {"".join(ch for ch in name.replace("（", "(").replace("）", ")").lower() if not ch.isspace()): name for name in ASPECT_NAMES}
TARGET_FIELD_CANDIDATES = [
    "target_aspect",
    "checklist_aspect",
    "target_aspect_name",
    "source_checklist_aspect",
    "source_target_aspect",
]
ATTACK_STRENGTH_FIELDS = [
    "attack_strength",
    "desired_attack_strength",
    "attackStrength",
    "source_attack_strength",
]
ANSWER_A_FIELDS = ["answer_a", "text_a", "response_a", "assistant_1", "candidate_a"]
ANSWER_B_FIELDS = ["answer_b", "text_b", "response_b", "assistant_2", "candidate_b"]
QUERY_FIELDS = ["query", "question", "prompt", "new_query", "generated_query"]
CHECKLIST_FIELDS = ["response_text","judge_response_text", "checklist", "checklist_json"]
THINK_SUFFIX = "-think"
THINKING_MODEL_HINTS = {
    "claude-sonnet-4": "claude",
    "claude-": "claude",
    "gemini": "gemini",
    "gpt-": "gpt",
    "gpt_": "gpt",
    "gpt ": "gpt",
    "gpt": "gpt",
    "o3": "gpt",
}
THINKING_MAX_BUDGETS = {
    "claude": 8096,
    "gemini": 4096,
}
THINKING_MIN_HEADROOM = 64
PREDICTION_FLUSH_INTERVAL = 20
DEFAULT_GPT_REASONING_EFFORT = os.environ.get("GPT_REASONING_EFFORT", "medium") or "medium"
ISOLATION_RATIO_EPS = 1e-9
QUESTION_ID_FIELDS = [
    "question_id",
    "sample_id",
    "id",
    "base_sample_id",
]

VIOLATION_RATIONALE_FIELDS = [
    "violation_rationale",
    "input__violation_rationale",
    "source_violation_rationale",
]
CHECKLIST_ASPECT_META_FIELDS = [
    "checklist_aspect",
    "source_checklist_aspect",
    "target_aspect",
]
PRESERVATION_NOTES_FIELDS = [
    "preservation_notes",
    "input__preservation_notes",
    "source_preservation_notes",
]
SWAP_ID_FIELDS = [
    "swap_id",
    "swapId",
]
SWAP_GROUP_FIELDS = [
    "swap_group_id",
    "swap_group",
    "base_sample_id",
]
SWAP_VARIANT_FIELDS = [
    "swap_variant",
    "variant",
]
SWAP_SUFFIX_VARIANTS = {
    "-orig": "original",
    "_orig": "original",
    "::orig": "original",
    "-swap": "swapped",
    "_swap": "swapped",
    "::swap": "swapped",
    "::swapped": "swapped",
}
LOGIC_PREFERENCE_PROMPT = """
You are a logic analysis expert. Based on the given **QUERY**, two answers (A and B), and a **logic evaluation checklist** containing eight logical dimensions, you will determine which side performs better logically in each dimension and provide detailed, targeted explanations. Finally, you will synthesize the performance across all eight dimensions to generate an overall judgment with a brief explanation.

——

## Task Objectives
You need to complete two levels of judgment:
1. **Dimension-by-dimension evaluation**: For each of the eight logical dimensions in the checklist, determine whether A or B performs better logically and explain the reasoning in English.
2. **Overall evaluation**: Synthesize the performance across all eight dimensions to determine the final winner (overall_winner) and provide a concise overall logical assessment.

——

## Analysis Steps
1. **Understand the checklist's logical signals**:  
   - The `"question"` field specifies the logical focus of each dimension (e.g., "Does it closely address the QUERY?", "Is the evidence relevant?", "Is the reasoning complete?").
   - The `"good_example"` and `"bad_example"` fields demonstrate typical characteristics of strong and weak performance, which can guide your assessment of A/B.
   - The `"span_hint"` field provides linguistic signals and logical cues (e.g., "therefore", "although...but...", "thus") to help you identify logical relationships.

2. **Compare A and B dimension by dimension**:  
   - Determine which side better meets the logical standards for that dimension.
   - Explain the specific reasons in English, describing the differences (do not directly quote the original text).
   - If both perform similarly or both are weak, clearly state this and explain the weaknesses.
   - Each dimension must include both a "winner" and an "explanation".

3. **Synthesize the overall judgment**:  
   - Based on the evaluation results across all eight dimensions, summarize which side performs better overall.
   - The `overall_explanation` must be a coherent English paragraph.
   - The first sentence should explain the overall winner and main reason, then use 1-2 sentences to supplement the overall logical performance characteristics (such as complete structure, coherent reasoning, sufficient evidence, etc.).

——

## Evaluation Principles
- Evaluate only **logical structure and argumentation quality**, not factual correctness or writing style.
- All explanations must be based on the checklist's logical focus and textual performance.
- Avoid vague expressions (such as "more reasonable", "better"); provide specific logical reasoning.
- Do not directly quote from the original answers; paraphrase the logical characteristics in your own words.
- The output must be a complete JSON object with consistent field naming and format.

——

## Eight Logical Dimensions (names must match exactly)
1. **Task alignment & claim clarity**  
2. **Global coherence**  
3. **Local coherence**  
4. **Evidence sufficiency & relevance**  
5. **Warrants & causal reasoning**  
6. **Qualifiers & counterpoints**  
7. **Internal consistency**  
8. **Concept introduction & logical transition**

——

## Input Format
[INPUT]
- QUERY  
<<<QUERY>>>

- TEXT A  
<<<TEXT_A>>>

- TEXT B  
<<<TEXT_B>>>

- CHECKLIST  
<<<CHECKLIST_JSON>>>

——

## Output Format (must strictly follow, only this one JSON object)
{
    "aspect_evaluations": {
        "Task alignment & claim clarity": {
            "winner": "A>B" | "B>A" | "both good" | "both bad",
            "explanation": "Summarize logical differences and reasons in English, do not quote original text"
        },
        "Global coherence": {
            "winner": "...",
            "explanation": "..."
        },
        "Local coherence": {
            "winner": "...",
            "explanation": "..."
        },
        "Evidence sufficiency & relevance": {
            "winner": "...",
            "explanation": "..."
        },
        "Warrants & causal reasoning": {
            "winner": "...",
            "explanation": "..."
        },
        "Qualifiers & counterpoints": {
            "winner": "...",
            "explanation": "..."
        },
        "Internal consistency": {
            "winner": "...",
            "explanation": "..."
        },
        "Concept introduction & logical transition": {
            "winner": "...",
            "explanation": "..."
        }
    },
    "overall_winner": "A>B" | "B>A" | "both good" | "both bad",
    "overall_explanation": "A concise English paragraph (2-3 sentences) summarizing the overall logical preference and main reasons (such as more complete structure, more coherent reasoning, more relevant evidence, more balanced qualifiers, more natural concept transitions, etc.), do not use lists or bullet points."
}

——

Please strictly follow all the above constraints and output only a JSON object that conforms to the key names and structure.
"""
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
    "都好": "both_good",
    "都不好": "both_bad",
}
SCORE_MAP = {
    "A": (1.0, 0.0),
    "B": (0.0, 1.0),
    "both_good": (0.5, 0.5),
    "both_bad": (0.5, 0.5),
}
_thread_local = threading.local()


class MissingAPIKeyError(RuntimeError):
    pass


class DebugSampleComplete(RuntimeError):
    """Raised to stop execution after printing debug samples."""


class _RateLimiter:
    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = time.monotonic()

    def acquire(self) -> None:
        if self.interval <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._next_time - now
                if wait <= 0:
                    self._next_time = now + self.interval
                    return
            if wait > 0:
                time.sleep(wait)


_RATE_LIMITERS: Dict[float, _RateLimiter] = {}
_RATE_LIMITERS_LOCK = threading.Lock()


def _get_rate_limiter(rate: float) -> Optional[_RateLimiter]:
    if rate is None or rate <= 0:
        return None
    key = float(rate)
    with _RATE_LIMITERS_LOCK:
        limiter = _RATE_LIMITERS.get(key)
        if limiter is None:
            limiter = _RateLimiter(rate)
            _RATE_LIMITERS[key] = limiter
    return limiter


@dataclass
class ModelConfig:
    alias: str
    model: str
    enable_think: bool = False

    @property
    def sanitized_alias(self) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", self.alias)
        return sanitized or "model"



def _strip_think_suffix(text: str) -> Tuple[str, bool]:
    trimmed = text.strip()
    lowered = trimmed.lower()
    if lowered.endswith(THINK_SUFFIX):
        return trimmed[: -len(THINK_SUFFIX)], True
    return trimmed, False


def _resolve_model_temperature(model_name: str, base_temperature: float) -> float:
    lowered = (model_name or "").lower()
    if "gpt" in lowered or "o3" in lowered:
        return 1.0
    return base_temperature


def _select_thinking_kind(model_name: str, enable_think: bool) -> Optional[str]:
    if not enable_think:
        return None
    lowered = (model_name or "").lower()
    for substring, kind in THINKING_MODEL_HINTS.items():
        if substring in lowered:
            return kind
    return None


def _build_thinking_extra_body(kind: Optional[str], max_tokens: Optional[int]) -> Optional[Dict[str, Any]]:
    if not kind:
        return None
    if kind == "gpt":
        return {"reasoning": {"effort": DEFAULT_GPT_REASONING_EFFORT}}
    if max_tokens is None:
        return None
    headroom = max(1, THINKING_MIN_HEADROOM)
    available_budget = max_tokens - headroom
    if available_budget <= 0:
        LOGGER.warning(
            "Skipping thinking mode because max_tokens=%s leaves no budget after %s headroom.",
            max_tokens,
            headroom,
        )
        return None
    if kind == "claude":
        budget = min(THINKING_MAX_BUDGETS["claude"], available_budget)
        if budget <= 0:
            LOGGER.warning("Skipping Claude thinking because available budget <= 0")
            return None
        return {"claude": {"thinking": {"type": "enabled", "budget_tokens": budget}}}
    if kind == "gemini":
        return {
            "google": {
                "thinking_config": {
                    "include_thoughts": True,
                },
                "thought_tag_marker": "think",
            }
        }
    return None


def _parse_model_configs(raw_values: Optional[Sequence[str]]) -> List[ModelConfig]:
    if not raw_values:
        raw_values = [DEFAULT_JUDGE_MODEL]
    flattened: List[str] = []
    for entry in raw_values:
        if isinstance(entry, (list, tuple, set)):
            for item in entry:
                if item is not None:
                    flattened.append(str(item))
        elif entry is not None:
            flattened.append(str(entry))
    raw_values = flattened
    if not raw_values:
        raw_values = [DEFAULT_JUDGE_MODEL]
    configs: List[ModelConfig] = []
    for entry in raw_values:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        alias: Optional[str] = None
        model_name: Optional[str] = None
        for sep in ("=", ":"):
            if sep in text:
                left, right = text.split(sep, 1)
                alias = left.strip() or None
                model_name = right.strip() or None
                break
        if model_name is None:
            model_name = text
        if alias is None:
            alias = model_name
        alias_lower = alias.lower()
        alias_think = alias_lower.endswith(THINK_SUFFIX)
        model_name, model_think = _strip_think_suffix(model_name)
        enable_think = alias_think or model_think
        configs.append(ModelConfig(alias=alias, model=model_name, enable_think=enable_think))
    if not configs:
        raise ValueError("No valid judge models provided")
    return configs


def _default_predictions_dir(input_path: Path) -> Path:
    stem = input_path.stem or "predictions"
    return input_path.with_name(f"{stem}_predictions")


def _resolve_predictions_path(
    *,
    args: argparse.Namespace,
    model: ModelConfig,
    input_path: Path,
) -> Path:
    directory = Path(args.predictions_dir) if args.predictions_dir else _default_predictions_dir(input_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{model.sanitized_alias}_predictions.jsonl"


@dataclass
class EvaluationTask:
    question_id: str
    query: str
    answer_a: str
    answer_b: str
    checklist: Any
    target_aspect: Optional[str]
    unique_key: str
    raw_record: Dict[str, Any]


@dataclass
class EvaluationResult:
    task: EvaluationTask
    response_text: str
    parsed_payload: Optional[Dict[str, Any]]


@dataclass
class AspectScore:
    winner: str
    clean_score: float
    adversarial_score: float
    delta: float
    damage: float
    explanation: Optional[str]


def _select_first_nonempty(row: Mapping[str, Any], candidates: Sequence[str]) -> str:
    for key in candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, str):
            text = str(value).strip()
            if text:
                return text
    return ""


def _build_unique_key(query: str, row: Mapping[str, Any]) -> str:
    components: List[str] = []
    query_text = query.strip()
    if query_text:
        components.append(query_text)
    swap_identifier = _select_first_nonempty(row, SWAP_ID_FIELDS)
    if swap_identifier:
        components.append(swap_identifier)
    question_identifier = _select_first_nonempty(row, QUESTION_ID_FIELDS)
    if question_identifier:
        components.append(question_identifier)
    rationale_text = _select_first_nonempty(row, VIOLATION_RATIONALE_FIELDS)
    if rationale_text:
        components.append(rationale_text)
    base = "|||".join(components)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    return WINNER_NORMALIZATION_MAP.get(lowered, text.upper() if text in {"A", "B"} else lowered)


def _scores_for_winner(label: str) -> Tuple[float, float]:
    winner = _normalize_label(label)
    scores = SCORE_MAP.get(winner)
    if scores is not None:
        return scores
    if winner in {"A", "B"}:
        return SCORE_MAP[winner]
    return 0.0, 0.0


def _split_swap_identifier(value: str) -> Tuple[str, Optional[str]]:
    text = value.strip()
    lowered = text.lower()
    for suffix, variant in SWAP_SUFFIX_VARIANTS.items():
        if lowered.endswith(suffix):
            return text[: -len(suffix)], variant
    return text, None


def _resolve_swap_metadata(row: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    swap_id = _select_first_nonempty(row, SWAP_ID_FIELDS)
    swap_group = _select_first_nonempty(row, SWAP_GROUP_FIELDS)
    swap_variant = _select_first_nonempty(row, SWAP_VARIANT_FIELDS)
    if swap_id:
        base_candidate, variant_candidate = _split_swap_identifier(swap_id)
        if base_candidate and not swap_group:
            swap_group = base_candidate
        if variant_candidate and not swap_variant:
            swap_variant = variant_candidate
    if swap_variant:
        swap_variant = swap_variant.strip().lower()
    if swap_group:
        swap_group = swap_group.strip()
    return swap_id or None, swap_group or None, swap_variant or None


def _infer_adversarial_side(swap_id: Optional[str], swap_variant: Optional[str]) -> str:
    variant = (swap_variant or "").strip().lower()
    if variant in {"swapped", "swap"}:
        return "B"
    if variant in {"original", "orig"}:
        return "A"

    identifier = (swap_id or "").strip().lower()
    if identifier:
        for suffix, label in SWAP_SUFFIX_VARIANTS.items():
            if identifier.endswith(suffix):
                return "B" if label == "swapped" else "A"
        if identifier.endswith("swap") or identifier.endswith("swapped"):
            return "B"
        if identifier.endswith("orig") or identifier.endswith("original"):
            return "A"

    return "A"


def _aspect_key(name: Any) -> str:
    if not isinstance(name, str):
        name = str(name or "")
    text = name.replace("（", "(").replace("）", ")")
    return "".join(ch for ch in text.lower() if not ch.isspace())


def _match_aspect_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = _aspect_key(value)
    if not key:
        return None
    if key in ASPECT_KEYS:
        return ASPECT_KEYS[key]
    for candidate_key, canonical in ASPECT_KEYS.items():
        if key in candidate_key or candidate_key in key:
            return canonical
    return None


def _safe_json_load(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_checklist(row: Mapping[str, Any]) -> Any:
    for field in CHECKLIST_FIELDS:
        value = row.get(field)
        if value in (None, ""):
            continue
        return value
    raise ValueError("Checklist payload not found in record")


def _resolve_target_aspect(row: Mapping[str, Any], checklist: Sequence[Mapping[str, Any]]) -> Optional[str]:
    for field in TARGET_FIELD_CANDIDATES:
        value = row.get(field)
        aspect = _match_aspect_name(value)
        if aspect:
            return aspect
    for item in checklist:
        if not isinstance(item, Mapping):
            continue
        aspect = _match_aspect_name(item.get("aspect"))
        if aspect:
            return aspect
    return None


def _resolve_attack_strength(row: Mapping[str, Any]) -> Optional[str]:
    for field in ATTACK_STRENGTH_FIELDS:
        value = row.get(field)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text.lower()
    return None


def _infer_input_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix_map = {
        ".jsonl": "jsonl",
        ".csv": "csv",
        ".tsv": "tsv",
    }
    suffix = path.suffix.lower()
    if suffix in suffix_map:
        return suffix_map[suffix]
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("{") or line.startswith("["):
                    return "jsonl"
                if "\t" in raw_line:
                    return "tsv"
                if "," in raw_line:
                    return "csv"
                break
    except UnicodeDecodeError:
        pass
    return "jsonl"


def _load_records(path: Path, *, input_format: str = "auto") -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    fmt = _infer_input_format(path, input_format)
    if fmt == "jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_num, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse JSON at line {line_num} in {path}: {exc}") from exc
                if isinstance(data, dict):
                    rows.append(data)
        return rows
    if fmt in {"csv", "tsv"}:
        delimiter = "," if fmt == "csv" else "\t"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [dict(row) for row in reader]
    raise ValueError(
        "Unsupported input file; use --input-format to select 'jsonl', 'csv', or 'tsv'."
    )


def _load_prediction_keys(path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num} in predictions file {path}: {exc}") from exc
            if isinstance(payload, Mapping):
                key = payload.get("unique_key")
                if isinstance(key, str) and key:
                    keys.add(key)
    return keys


def _load_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_num} in predictions file {path}: {exc}"
                ) from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _partition_predictions_for_resume(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Set[str], Set[str]]:
    rows = _load_prediction_rows(path)
    _upgrade_prediction_rows_with_damage(path, rows)
    latest_by_key: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row.get("unique_key")
        if isinstance(key, str) and key:
            latest_by_key[key] = row
    completed_keys: Set[str] = set()
    retry_keys: Set[str] = set()
    for key, latest_row in latest_by_key.items():
        if latest_row.get("targeted_damage") is None:
            retry_keys.add(key)
        else:
            completed_keys.add(key)
    return rows, latest_by_key, completed_keys, retry_keys


def _rewrite_predictions_excluding_keys(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    drop_keys: Set[str],
) -> None:
    if not drop_keys:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            key = row.get("unique_key")
            if isinstance(key, str) and key in drop_keys:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _rewrite_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _compute_damage_from_entry(entry: Mapping[str, Any]) -> Optional[float]:
    if not isinstance(entry, Mapping):
        return None
    clean = entry.get("clean_score")
    adv = entry.get("adversarial_score")
    try:
        clean_val = float(clean)
        adv_val = float(adv)
    except (TypeError, ValueError):
        return None
    damage = max(0.0, adv_val - clean_val)
    return damage


def _is_adv_nonworse(clean: Any, adversarial: Any) -> Optional[bool]:
    try:
        clean_val = float(clean)
        adv_val = float(adversarial)
    except (TypeError, ValueError):
        return None
    return adv_val >= clean_val


def _derive_damage_fields(row: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    target_aspect = row.get("target_aspect")
    aspect_scores = row.get("aspect_scores")
    if not isinstance(target_aspect, str) or not isinstance(aspect_scores, Mapping):
        return None, None
    target_entry = aspect_scores.get(target_aspect)
    targeted_damage = _compute_damage_from_entry(target_entry)
    other_damages: List[float] = []
    for name, entry in aspect_scores.items():
        if name == target_aspect:
            continue
        damage_value = _compute_damage_from_entry(entry)
        if damage_value is not None:
            other_damages.append(damage_value)
    non_target_damage = None
    if targeted_damage is not None:
        if other_damages:
            non_target_damage = sum(other_damages) / len(other_damages)
        else:
            non_target_damage = 0.0
    return targeted_damage, non_target_damage


def _infer_attack_success_from_row(row: Mapping[str, Any]) -> Optional[bool]:
    overall_winner = row.get("overall_winner")
    normalized = _normalize_label(overall_winner)
    if normalized in {"A", "B", "both_good", "both_bad"}:
        return normalized != "A"
    return None


def _upgrade_prediction_rows_with_damage(path: Path, rows: Sequence[Dict[str, Any]]) -> bool:
    updated = False
    for row in rows:
        aspect_scores = row.get("aspect_scores")
        if isinstance(aspect_scores, Mapping):
            for entry in aspect_scores.values():
                if isinstance(entry, dict):
                    damage_value = _compute_damage_from_entry(entry)
                    if damage_value is not None and entry.get("damage") != damage_value:
                        entry["damage"] = damage_value
                        updated = True
        targeted_damage, non_target_damage = _derive_damage_fields(row)
        if targeted_damage is not None and row.get("targeted_damage") != targeted_damage:
            row["targeted_damage"] = targeted_damage
            updated = True
        if non_target_damage is not None and row.get("non_target_damage") != non_target_damage:
            row["non_target_damage"] = non_target_damage
            updated = True
        target_aspect = row.get("target_aspect")
        target_entry = aspect_scores.get(target_aspect) if isinstance(aspect_scores, Mapping) else None
        adv_nonworse = None
        if isinstance(target_entry, Mapping):
            adv_nonworse = _is_adv_nonworse(target_entry.get("clean_score"), target_entry.get("adversarial_score"))
        if adv_nonworse is not None:
            if row.get("attack_success") != adv_nonworse:
                row["attack_success"] = adv_nonworse
                updated = True
    if updated:
        _rewrite_predictions(path, rows)
    return updated


def _backfill_prediction_field(
    *,
    path: Path,
    rows: Sequence[Dict[str, Any]],
    field_name: str,
    value_lookup: Mapping[str, Optional[str]],
) -> bool:
    updated = False
    for row in rows:
        current_value = row.get(field_name)
        if current_value not in (None, ""):
            continue
        key = row.get("unique_key")
        if not isinstance(key, str):
            continue
        lookup_value = value_lookup.get(key)
        if lookup_value in (None, ""):
            continue
        row[field_name] = lookup_value
        updated = True
    if updated:
        _rewrite_predictions(path, rows)
    return updated


def _print_judge_response_text(records: Sequence[Mapping[str, Any]], limit: Optional[int]) -> None:
    printed = 0
    cap = None if limit is None or limit <= 0 else limit
    for idx, row in enumerate(records):
        value = row.get("judge_response_text")
        payload = {
            "record_index": idx,
            "question_id": _select_first_nonempty(row, ["question_id", "sample_id", "id"]) or f"sample-{idx}",
            "judge_response_text": value,
        }
        print(json.dumps({"peek_judge_response_text": payload}, ensure_ascii=False))
        printed += 1
        if cap is not None and printed >= cap:
            break
    if printed == 0:
        print(
            json.dumps(
                {
                    "peek_judge_response_text": {
                        "message": "No rows contained the judge_response_text field."
                    }
                },
                ensure_ascii=False,
            )
        )


def _build_prompt(task: EvaluationTask) -> str:
    checklist_text = _format_checklist_payload(task.checklist)
    return (
        LOGIC_PREFERENCE_PROMPT
        .replace("<<<QUERY>>>", task.query)
        .replace("<<<TEXT_A>>>", task.answer_a)
        .replace("<<<TEXT_B>>>", task.answer_b)
        .replace("<<<CHECKLIST_JSON>>>", checklist_text)
        .strip()
    )


def _format_checklist_payload(payload: Any) -> str:
    if payload is None:
        return "[]"
    if isinstance(payload, str):
        return payload.strip() or "[]"
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        text = str(payload)
        return text.strip() or "[]"


class JudgeClient:
    def __init__(
        self,
        model: str,
        *,
        temperature: float,
        max_tokens: int,
        max_attempts: int,
        requests_per_second: float,
        request_timeout: Optional[int],
        api_key_env: str,
        base_url: Optional[str],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        if OpenAI is None:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError("The 'openai' package is required to run this script.")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise MissingAPIKeyError(f"API key not found in environment variable {api_key_env}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout or DEFAULT_REQUEST_TIMEOUT
        self.rate_limiter = _get_rate_limiter(requests_per_second)
        self.extra_body = extra_body

    def score(self, task: EvaluationTask) -> EvaluationResult:
        messages = [{"role": "user", "content": _build_prompt(task)}]
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": messages,
                    "stream": False,
                }
                if self.max_tokens is not None:
                    kwargs["max_completion_tokens"] = self.max_tokens
                if self.request_timeout is not None:
                    kwargs["timeout"] = self.request_timeout
                if self.extra_body is not None:
                    kwargs["extra_body"] = self.extra_body
                completion = self.client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content if completion.choices else ""
                parsed = _extract_json(content)
                if not isinstance(parsed, dict):
                    raise RuntimeError(
                        f"LLM judge response is not a JSON object (raw={content})"
                    )
                return EvaluationResult(task=task, response_text=content or "", parsed_payload=parsed)
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                time.sleep(min(3 * attempt, 10))
        raise RuntimeError(f"Judge request failed after {self.max_attempts} attempts: {last_error}")


def _strip_code_fence_block(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    closing = stripped.rfind("```")
    if closing <= 0:
        return stripped
    inner = stripped[3:closing]
    newline = inner.find("\n")
    if newline != -1:
        inner = inner[newline + 1 :]
    return inner.strip()


def _repair_unescaped_quotes(payload: str) -> str:
    if not payload:
        return payload
    result: List[str] = []
    length = len(payload)
    i = 0
    while i < length:
        ch = payload[i]
        if ch == '"':
            backslash_count = 0
            j = i - 1
            while j >= 0 and payload[j] == "\\":
                backslash_count += 1
                j -= 1
            if backslash_count % 2 == 1:
                result.append(ch)
                i += 1
                continue
            prev = j
            while prev >= 0 and payload[prev].isspace():
                prev -= 1
            prev_char = payload[prev] if prev >= 0 else ""
            nxt = i + 1
            while nxt < length and payload[nxt].isspace():
                nxt += 1
            next_char = payload[nxt] if nxt < length else ""
            if prev_char in {"", ":", "{", "[", ","}:
                result.append(ch)
            elif next_char in {"", ",", "}", "]", ":"}:
                result.append(ch)
            else:
                result.append("\\\"")
                i += 1
                continue
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _try_json_load(payload: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = _strip_code_fence_block(text.strip())
    parsed = _try_json_load(text)
    if parsed is not None:
        return parsed
    relaxed = _repair_unescaped_quotes(text)
    if relaxed != text:
        parsed = _try_json_load(relaxed)
        if parsed is not None:
            return parsed
        text = relaxed
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        snippet = text[start : end + 1]
        parsed = _try_json_load(snippet)
        if parsed is not None:
            return parsed
        repaired_snippet = _escape_unquoted_inner_quotes(snippet)
        if repaired_snippet != snippet:
            parsed = _try_json_load(repaired_snippet)
            if parsed is not None:
                return parsed
    repaired_full = _escape_unquoted_inner_quotes(text)
    if repaired_full != text:
        parsed = _try_json_load(repaired_full)
        if parsed is not None:
            return parsed
    return None


def _escape_unquoted_inner_quotes(text: str) -> str:
    chars: List[str] = []
    in_string = False
    escape = False
    length = len(text)
    json_terminators = {",", "}", "]", ":", " ", "\n", "\r", "\t"}
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                chars.append(ch)
                escape = False
                continue
            if ch == "\\":
                chars.append(ch)
                escape = True
                continue
            if ch == '"':
                next_char = text[idx + 1] if idx + 1 < length else ""
                if next_char and next_char not in json_terminators:
                    chars.extend(["\\", '"'])
                    continue
                chars.append(ch)
                in_string = False
                continue
            chars.append(ch)
            continue
        chars.append(ch)
        if ch == '"':
            in_string = True
            escape = False
    return "".join(chars)


def _build_tasks(
    records: Sequence[Dict[str, Any]],
    limit: Optional[int],
    seed: Optional[int],
    *,
    debug_filter: bool = False,
    debug_filter_limit: int = 20,
) -> List[EvaluationTask]:
    tasks: List[EvaluationTask] = []
    dropped_stats: Counter[str] = Counter()
    debug_samples: List[Dict[str, Any]] = []
    debug_cap = max(0, debug_filter_limit)

    def _log_drop(reason: str, row: Mapping[str, Any], row_index: int, query: str, answer_a: str, answer_b: str, checklist_ok: bool) -> None:
        dropped_stats[reason] += 1
        if debug_filter and len(debug_samples) < debug_cap:
            debug_samples.append(
                {
                    "index": row_index,
                    "reason": reason,
                    "question_id": _select_first_nonempty(row, ["question_id", "sample_id", "id"]) or f"sample-{row_index}",
                    "has_query": bool(query.strip()),
                    "has_answer_a": bool(answer_a.strip()),
                    "has_answer_b": bool(answer_b.strip()),
                    "has_checklist": checklist_ok,
                    "sample_keys": list(row.keys())[:12],
                }
            )

    for idx, row in enumerate(records):
        checklist_ok = False
        checklist_payload: Any
        try:
            checklist_payload = _extract_checklist(row)
            checklist_ok = True
        except ValueError:
            _log_drop("missing_checklist", row, idx, "", "", "", False)
            continue
        query = _select_first_nonempty(row, QUERY_FIELDS)
        answer_a = _select_first_nonempty(row, ANSWER_A_FIELDS)
        answer_b = _select_first_nonempty(row, ANSWER_B_FIELDS)
        if not query:
            _log_drop("missing_query", row, idx, query, answer_a, answer_b, checklist_ok)
            continue
        if not answer_a:
            _log_drop("missing_answer_a", row, idx, query, answer_a, answer_b, checklist_ok)
            continue
        if not answer_b:
            _log_drop("missing_answer_b", row, idx, query, answer_a, answer_b, checklist_ok)
            continue
        unique_key = _build_unique_key(query, row)
        checklist_for_aspect: Sequence[Mapping[str, Any]]
        if isinstance(checklist_payload, Sequence) and not isinstance(checklist_payload, (str, bytes)):
            checklist_for_aspect = checklist_payload  # type: ignore[assignment]
        else:
            checklist_for_aspect = []
        target_aspect = _resolve_target_aspect(row, checklist_for_aspect)
        question_id = _select_first_nonempty(row, ["question_id", "sample_id", "id"]) or f"sample-{idx}"
        tasks.append(
            EvaluationTask(
                question_id=question_id,
                query=query,
                answer_a=answer_a,
                answer_b=answer_b,
                checklist=checklist_payload,
                target_aspect=target_aspect,
                unique_key=unique_key,
                raw_record=dict(row),
            )
        )

    if seed is not None:
        random.Random(seed).shuffle(tasks)
    if limit is not None:
        tasks = tasks[:limit]

    total_dropped = sum(dropped_stats.values())
    if debug_filter:
        print(
            json.dumps(
                {
                    "records": len(records),
                    "runnable_tasks": len(tasks),
                    "dropped_records": total_dropped,
                    "drop_reasons": dict(dropped_stats),
                },
                ensure_ascii=False,
            )
        )
        for sample in debug_samples:
            print(
                json.dumps(
                    {
                        "debug": "dropped_record",
                        **sample,
                    },
                    ensure_ascii=False,
                )
            )
    elif not tasks and total_dropped:
        print(
            json.dumps(
                {
                    "records": len(records),
                    "runnable_tasks": 0,
                    "dropped_records": total_dropped,
                    "drop_reasons": dict(dropped_stats),
                    "hint": "All rows were filtered. Re-run with --debug-filter to print sample-level details.",
                },
                ensure_ascii=False,
            )
        )

    return tasks


def _score_aspects(parsed: Mapping[str, Any], adversarial_side: str) -> Dict[str, AspectScore]:
    aspect_payload = parsed.get("aspect_evaluations")
    if not isinstance(aspect_payload, Mapping):
        return {}
    adv_is_a = (adversarial_side or "A").strip().upper() == "A"
    scores: Dict[str, AspectScore] = {}
    for raw_name, payload in aspect_payload.items():
        if not isinstance(payload, Mapping):
            continue
        canonical_name = _match_aspect_name(raw_name)
        if not canonical_name:
            continue
        winner = payload.get("winner")
        score_a, score_b = _scores_for_winner(winner)
        if adv_is_a:
            clean_score, adv_score = score_b, score_a
        else:
            clean_score, adv_score = score_a, score_b
        damage = max(0.0, adv_score - clean_score)
        scores[canonical_name] = AspectScore(
            winner=_normalize_label(winner) or "",
            clean_score=clean_score,
            adversarial_score=adv_score,
            delta=clean_score - adv_score,
            damage=damage,
            explanation=str(payload.get("explanation", "")) if "explanation" in payload else None,
        )
    return scores


def _summarize_sample(result: EvaluationResult) -> Dict[str, Any]:
    parsed = result.parsed_payload or {}
    target_aspect = result.task.target_aspect
    raw_record = result.task.raw_record
    attack_strength = _resolve_attack_strength(raw_record)
    swap_id, swap_group_id, swap_variant = _resolve_swap_metadata(raw_record)
    adversarial_side = _infer_adversarial_side(swap_id, swap_variant)
    aspect_scores = _score_aspects(parsed, adversarial_side)
    violation_rationale = _select_first_nonempty(raw_record, VIOLATION_RATIONALE_FIELDS) or None
    checklist_aspect = _select_first_nonempty(raw_record, CHECKLIST_ASPECT_META_FIELDS) or None
    preservation_notes = _select_first_nonempty(raw_record, PRESERVATION_NOTES_FIELDS) or None
    targeted_damage: Optional[float] = None
    non_target_damage: Optional[float] = None
    attack_success: Optional[bool] = None
    if target_aspect and target_aspect in aspect_scores:
        target_score = aspect_scores[target_aspect]
        targeted_damage = target_score.damage
        other_damages = [score.damage for name, score in aspect_scores.items() if name != target_aspect]
        non_target_damage = sum(other_damages) / len(other_damages) if other_damages else 0.0
        adv_nonworse = _is_adv_nonworse(target_score.clean_score, target_score.adversarial_score)
        if adv_nonworse is not None:
            attack_success = adv_nonworse
    elif target_aspect is None:
        target_aspect = None
    return {
        "question_id": result.task.question_id,
        "unique_key": result.task.unique_key,
        "target_aspect": target_aspect,
        "attack_strength": attack_strength,
        "swap_id": swap_id,
        "swap_group_id": swap_group_id,
        "swap_variant": swap_variant,
    "adversarial_side": adversarial_side,
        "violation_rationale": violation_rationale,
        "checklist_aspect": checklist_aspect,
        "preservation_notes": preservation_notes,
        "aspect_scores": {
            name: {
                "winner": score.winner,
                "clean_score": score.clean_score,
                "adversarial_score": score.adversarial_score,
                "delta": score.delta,
                "damage": score.damage,
                "explanation": score.explanation,
            }
            for name, score in aspect_scores.items()
        },
        "targeted_damage": targeted_damage,
        "non_target_damage": non_target_damage,
        "attack_success": attack_success,
        "overall_winner": parsed.get("overall_winner"),
        "overall_explanation": parsed.get("overall_explanation"),
        "llm_response": result.response_text,
    }


def _is_high_strength_success(summary: Mapping[str, Any]) -> bool:
    strength = str(summary.get("attack_strength") or "").strip().lower()
    return strength == "high" and bool(summary.get("attack_success"))


def _print_debug_high_strength_sample(summary: Mapping[str, Any], task: EvaluationTask) -> None:
    target_aspect = summary.get("target_aspect")
    aspect_scores = summary.get("aspect_scores", {})
    target_scores = aspect_scores.get(target_aspect, {}) if isinstance(aspect_scores, Mapping) else {}
    adversarial_side = (summary.get("adversarial_side") or "A").strip().upper()
    if adversarial_side == "A":
        clean_answer = task.answer_b
        adversarial_answer = task.answer_a
    else:
        clean_answer = task.answer_a
        adversarial_answer = task.answer_b
    payload = {
        "question_id": summary.get("question_id"),
        "unique_key": summary.get("unique_key"),
        "target_aspect": target_aspect,
        "attack_strength": summary.get("attack_strength"),
        "targeted_damage": summary.get("targeted_damage"),
        "non_target_damage": summary.get("non_target_damage"),
        "attack_success": summary.get("attack_success"),
        "overall_winner": summary.get("overall_winner"),
        "overall_explanation": summary.get("overall_explanation"),
        "target_aspect_scores": target_scores,
        "query": task.query,
        "clean_answer": clean_answer,
        "adversarial_answer": adversarial_answer,
    }
    print(json.dumps({"debug_high_strength_sample": payload}, ensure_ascii=False, indent=2))


def _compute_metrics_totals(sample_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [summary for summary in sample_summaries if summary.get("targeted_damage") is not None]
    if not eligible:
        base_metrics = {
            "num_samples": len(sample_summaries),
            "num_valid_samples": 0,
            "attack_success_rate": 0.0,
            "targeted_damage": 0.0,
            "non_target_damage": 0.0,
            "isolation_ratio": 0.0,
        }
        base_metrics.update(_compute_swap_group_metrics(sample_summaries))
        return base_metrics
    num_valid = len(eligible)
    asr = sum(1 for item in eligible if item.get("attack_success")) / num_valid
    avg_targeted_damage = sum(float(item.get("targeted_damage") or 0.0) for item in eligible) / num_valid
    avg_non_target_damage = sum(float(item.get("non_target_damage") or 0.0) for item in eligible) / num_valid
    isolation_ratio = avg_targeted_damage / (
        avg_targeted_damage + avg_non_target_damage + ISOLATION_RATIO_EPS
    )
    base_metrics = {
        "num_samples": len(sample_summaries),
        "num_valid_samples": num_valid,
        "attack_success_rate": asr,
        "targeted_damage": avg_targeted_damage,
        "non_target_damage": avg_non_target_damage,
        "isolation_ratio": isolation_ratio,
    }
    base_metrics.update(_compute_swap_group_metrics(sample_summaries))
    return base_metrics


def _compute_swap_group_metrics(sample_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for summary in sample_summaries:
        group_id = summary.get("swap_group_id")
        if not group_id:
            continue
        key = str(group_id)
        grouped.setdefault(key, []).append(summary)
    if not grouped:
        return {
            "swap_group_total": 0,
            "swap_group_with_valid": 0,
            "swap_group_with_pairs": 0,
            "swap_attack_success_weak_count": 0,
            "swap_attack_success_robust_count": 0,
            "swap_attack_success_rate_weak": 0.0,
            "swap_attack_success_rate_robust": 0.0,
        }
    weak_denominator = 0
    weak_success = 0
    robust_denominator = 0
    robust_success = 0
    for rows in grouped.values():
        valid_rows = [row for row in rows if row.get("attack_success") is not None]
        if not valid_rows:
            continue
        weak_denominator += 1
        if any(bool(row.get("attack_success")) for row in valid_rows):
            weak_success += 1
        if len(valid_rows) >= 2:
            robust_denominator += 1
            if all(bool(row.get("attack_success")) for row in valid_rows):
                robust_success += 1
    return {
        "swap_group_total": len(grouped),
        "swap_group_with_valid": weak_denominator,
        "swap_group_with_pairs": robust_denominator,
        "swap_attack_success_weak_count": weak_success,
        "swap_attack_success_robust_count": robust_success,
        "swap_attack_success_rate_weak": (weak_success / weak_denominator) if weak_denominator else 0.0,
        "swap_attack_success_rate_robust": (robust_success / robust_denominator) if robust_denominator else 0.0,
    }


def _aggregate_metrics(sample_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return _compute_metrics_totals(sample_summaries)


def _aggregate_metrics_by_field(
    sample_summaries: Sequence[Dict[str, Any]],
    field_name: str,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for summary in sample_summaries:
        value = summary.get(field_name)
        if value in (None, ""):
            continue
        key = str(value)
        grouped.setdefault(key, []).append(summary)
    return {key: _compute_metrics_totals(rows) for key, rows in grouped.items()}


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


class _JsonlBufferedWriter:
    def __init__(self, path: Path, *, flush_every: int = 20, append: bool = False) -> None:
        self.path = Path(path)
        self.flush_every = max(1, flush_every)
        mode = "a" if append else "w"
        self._handle = self.path.open(mode, encoding="utf-8")
        self._buffer: List[Mapping[str, Any]] = []

    def write(self, row: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("Writer is closed")
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._handle is None or not self._buffer:
            return
        for row in self._buffer:
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()
        self._buffer.clear()

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self.flush()
        finally:
            self._handle.close()
            self._handle = None


def _run_model_evaluations(
    *,
    tasks: Sequence[EvaluationTask],
    model_config: ModelConfig,
    base_judge_params: Mapping[str, Any],
    num_workers: int,
    progress_factory: Any,
    base_temperature: float,
    existing_keys: Optional[Set[str]] = None,
    predictions_writer: Optional[_JsonlBufferedWriter] = None,
    debug_high_strength_mode: bool = False,
    debug_sample_limit: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int, Dict[str, Any]]:
    judge_params = dict(base_judge_params)
    judge_params["model"] = model_config.model
    temperature = _resolve_model_temperature(model_config.model, base_judge_params["temperature"])
    judge_params["temperature"] = temperature
    thinking_kind = _select_thinking_kind(model_config.model, model_config.enable_think)
    extra_body = _build_thinking_extra_body(thinking_kind, judge_params.get("max_tokens"))
    judge_params["extra_body"] = extra_body
    total_tasks = len(tasks)
    resume_skipped = 0
    if existing_keys:
        filtered_tasks = [task for task in tasks if task.unique_key not in existing_keys]
        resume_skipped = total_tasks - len(filtered_tasks)
        print(
            json.dumps(
                {
                    "model": model_config.alias,
                    "resume_total_tasks": total_tasks,
                    "resume_skipped": resume_skipped,
                    "resume_pending": len(filtered_tasks),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        filtered_tasks = list(tasks)
    if not filtered_tasks:
        run_meta = {
            "temperature": temperature,
            "thinking_kind": thinking_kind,
            "think_payload": extra_body,
            "skipped_due_to_resume": bool(existing_keys),
            "resume_skipped": resume_skipped,
            "resume_total_tasks": total_tasks,
        }
        return [], _aggregate_metrics([]), 0, run_meta
    errors = 0
    failed_examples: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    extra_body_fingerprint = json.dumps(extra_body, ensure_ascii=False, sort_keys=True) if extra_body else None

    def _score_task(task: EvaluationTask) -> EvaluationResult:
        client = getattr(_thread_local, "judge_client", None)
        client_key = getattr(_thread_local, "judge_client_key", None)
        key = (
            judge_params.get("model"),
            judge_params.get("temperature"),
            judge_params.get("max_tokens"),
            judge_params.get("max_attempts"),
            judge_params.get("requests_per_second"),
            judge_params.get("request_timeout"),
            judge_params.get("api_key_env"),
            judge_params.get("base_url"),
            extra_body_fingerprint,
        )
        if client is None or client_key != key:
            client = JudgeClient(**judge_params)
            _thread_local.judge_client = client
            _thread_local.judge_client_key = key
        return client.score(task)

    if debug_high_strength_mode:
        progress = progress_factory(
            total=len(filtered_tasks),
            desc=f"DebugScoring[{model_config.alias}]",
            dynamic_ncols=True,
            unit="task",
        )
        debug_hits = 0
        try:
            for task in filtered_tasks:
                try:
                    result = _score_task(task)
                    summary = _summarize_sample(result)
                    summaries.append(summary)
                    if predictions_writer is not None:
                        predictions_writer.write(summary)
                    if _is_high_strength_success(summary):
                        _print_debug_high_strength_sample(summary, task)
                        debug_hits += 1
                        if debug_hits >= max(1, debug_sample_limit):
                            raise DebugSampleComplete()
                except Exception as exc:  # pylint: disable=broad-except
                    errors += 1
                    print(f"[WARN] Failed to score {task.question_id}: {exc}", file=sys.stderr)
                    if len(failed_examples) < 2:
                        example_payload = {
                            "question_id": task.question_id,
                            "unique_key": task.unique_key,
                            "target_aspect": task.target_aspect,
                            "error": str(exc),
                        }
                        failed_examples.append(example_payload)
                        print(
                            json.dumps(
                                {
                                    "model": model_config.alias,
                                    "failed_example_peek": [example_payload],
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                finally:
                    progress.update(1)
        finally:
            progress.close()
        print(
            json.dumps(
                {
                    "model": model_config.alias,
                    "debug_message": "Exhausted tasks without finding enough high-strength attack-success samples.",
                    "samples_found": debug_hits,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise DebugSampleComplete()

    progress = progress_factory(
        total=len(filtered_tasks),
        desc=f"Scoring[{model_config.alias}]",
        dynamic_ncols=True,
        unit="task",
    )
    try:
        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            future_map = {executor.submit(_score_task, task): task for task in filtered_tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    result = future.result()
                    summary = _summarize_sample(result)
                    summaries.append(summary)
                    if predictions_writer is not None:
                        predictions_writer.write(summary)
                except Exception as exc:  # pylint: disable=broad-except
                    errors += 1
                    print(f"[WARN] Failed to score {task.question_id}: {exc}", file=sys.stderr)
                    if len(failed_examples) < 2:
                        example_payload = {
                            "question_id": task.question_id,
                            "unique_key": task.unique_key,
                            "target_aspect": task.target_aspect,
                            "error": str(exc),
                        }
                        failed_examples.append(example_payload)
                        print(
                            json.dumps(
                                {
                                    "model": model_config.alias,
                                    "failed_example_peek": [example_payload],
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                finally:
                    progress.update(1)
    finally:
        progress.close()

    # if not summaries:
    #     raise RuntimeError(f"All LLM judge calls failed for model {model_config.alias}; cannot compute metrics")
    metrics = _aggregate_metrics(summaries)
    run_meta = {
        "temperature": temperature,
        "thinking_kind": thinking_kind,
        "think_payload": extra_body,
        "failed_examples": failed_examples,
        "resume_skipped": resume_skipped,
        "resume_total_tasks": total_tasks,
    }
    return summaries, metrics, errors, run_meta


def _is_missing_prompt_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if value.__class__.__name__ == "NAType":  # pandas.NA without importing pandas at runtime
        return True
    return False



def _evaluate_model_config(
    *,
    config: ModelConfig,
    tasks: Sequence[EvaluationTask],
    args: argparse.Namespace,
    input_path: Path,
    base_judge_params: Mapping[str, Any],
    progress_factory: Any,
    base_temperature: float,
    debug_high_strength_mode: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], int]:
    task_attack_strengths: Dict[str, str] = {}
    for task in tasks:
        strength = _resolve_attack_strength(task.raw_record)
        if strength:
            task_attack_strengths[task.unique_key] = strength
    predictions_path = _resolve_predictions_path(
        args=args,
        model=config,
        input_path=input_path,
    )
    existing_keys: Optional[Set[str]] = None
    append_predictions = False
    resume_retry_keys: Set[str] = set()
    resume_retry_peek: List[Dict[str, Any]] = []
    if args.resume:
        rows_for_resume, latest_rows_by_key, completed_keys, retry_keys = _partition_predictions_for_resume(
            predictions_path
        )
        if retry_keys:
            _rewrite_predictions_excluding_keys(
                predictions_path,
                rows=rows_for_resume,
                drop_keys=retry_keys,
            )
            resume_retry_keys = retry_keys
            for key in list(retry_keys)[:3]:
                latest_row = latest_rows_by_key.get(key, {})
                resume_retry_peek.append(
                    {
                        "question_id": latest_row.get("question_id"),
                        "unique_key": key,
                        "target_aspect": latest_row.get("target_aspect"),
                        "targeted_damage": latest_row.get("targeted_damage"),
                    }
                )
            print(
                json.dumps(
                    {
                        "model": config.alias,
                        "resume_retried_samples": len(retry_keys),
                        "resume_retry_peek": resume_retry_peek,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        existing_keys = completed_keys
        append_predictions = True
    else:
        try:
            predictions_path.unlink()
        except FileNotFoundError:
            pass
    writer = _JsonlBufferedWriter(
        predictions_path,
        flush_every=PREDICTION_FLUSH_INTERVAL,
        append=append_predictions,
    )
    try:
        summaries, run_metrics, errors, run_meta = _run_model_evaluations(
            tasks=tasks,
            model_config=config,
            base_judge_params=base_judge_params,
            num_workers=args.num_workers,
            progress_factory=progress_factory,
            base_temperature=base_temperature,
            existing_keys=existing_keys,
            predictions_writer=writer,
            debug_high_strength_mode=debug_high_strength_mode,
            debug_sample_limit=2,
        )
    finally:
        writer.close()
    run_meta["resume_retry_count"] = len(resume_retry_keys)
    run_meta["resume_retry_peek"] = resume_retry_peek
    all_prediction_rows = _load_prediction_rows(predictions_path)
    _upgrade_prediction_rows_with_damage(predictions_path, all_prediction_rows)
    attack_strength_backfilled = _backfill_prediction_field(
        path=predictions_path,
        rows=all_prediction_rows,
        field_name="attack_strength",
        value_lookup=task_attack_strengths,
    )
    final_metrics = _aggregate_metrics(all_prediction_rows)
    metrics_by_attack_strength = _aggregate_metrics_by_field(all_prediction_rows, "attack_strength")
    metrics_by_target_aspect = _aggregate_metrics_by_field(all_prediction_rows, "target_aspect")
    aggregated_entry = {
        "api_model": config.model,
        "prediction_path": str(predictions_path),
        "num_failed_requests": errors,
        "metrics": final_metrics,
        "metrics_by_attack_strength": metrics_by_attack_strength,
        "metrics_by_target_aspect": metrics_by_target_aspect,
        "think_requested": config.enable_think,
        "thinking_kind": run_meta.get("thinking_kind"),
        "temperature_used": run_meta.get("temperature"),
        "skipped_due_to_resume": run_meta.get("skipped_due_to_resume", False),
        "resume_skipped": run_meta.get("resume_skipped", 0),
        "resume_total_tasks": run_meta.get("resume_total_tasks", len(tasks)),
        "resume_retry_count": run_meta.get("resume_retry_count", 0),
        "resume_retry_peek": run_meta.get("resume_retry_peek", []),
        "failed_examples": run_meta.get("failed_examples", []),
        "latest_run_metrics": run_metrics,
    }
    print_payload = {
        "model": config.alias,
        "api_model": config.model,
        "prediction_path": str(predictions_path),
        "num_failed_requests": errors,
        "metrics": final_metrics,
    "metrics_by_attack_strength": metrics_by_attack_strength,
    "metrics_by_target_aspect": metrics_by_target_aspect,
        "think_requested": config.enable_think,
        "thinking_kind": run_meta.get("thinking_kind"),
        "temperature_used": run_meta.get("temperature"),
        "skipped_due_to_resume": run_meta.get("skipped_due_to_resume", False),
        "resume_skipped": run_meta.get("resume_skipped", 0),
        "resume_total_tasks": run_meta.get("resume_total_tasks", len(tasks)),
        "resume_retry_count": run_meta.get("resume_retry_count", 0),
        "latest_run_metrics": run_metrics,
    }
    failed_examples = run_meta.get("failed_examples") or []
    return config.alias, aggregated_entry, print_payload, failed_examples, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge and compute targeted attack metrics")
    parser.add_argument("--input", required=True, help="Path to JSONL or CSV file containing prompts + checklist")
    parser.add_argument(
        "--input-format",
        choices=["auto", "jsonl", "csv", "tsv"],
        default="auto",
        help="Format of --input file. Default auto detects via extension/content.",
    )
    parser.add_argument(
        "--judge-model",
        dest="judge_models",
        action="append",
        nargs="+",
        help=(
            "Judge model identifier. Repeat this flag for multiple models. "
            "Use alias=model_name to customize display alias; defaults to the model string."
        ),
        default=None,
    )
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=30760)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--requests-per-second", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predictions-dir",
        help=(
            "Directory to store per-model prediction JSONL files (default: '<input>_predictions')."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing per-model prediction files found in --predictions-dir.",
    )
    parser.add_argument("--metrics-json", help="Optional path to dump aggregated metrics JSON")
    parser.add_argument(
        "--debug-filter",
        action="store_true",
        help="Print stats / samples for rows filtered before scoring",
    )
    parser.add_argument(
        "--debug-filter-limit",
        type=int,
        default=20,
        help="How many filtered rows to print when --debug-filter is on",
    )
    parser.add_argument(
        "--show-judge-response-text",
        action="store_true",
        help="Print the value of judge_response_text per record (respects --show-judge-response-limit).",
    )
    parser.add_argument(
        "--show-judge-response-limit",
        type=int,
        default=5,
    help="Maximum rows to print when showing judge_response_text (<=0 prints all rows).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Debug mode: stop after printing two attack_strength=='high' samples whose attack_success is True."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(Path(args.input), input_format=args.input_format)
    if not records:
        raise SystemExit("No rows found in input file")
    if args.show_judge_response_text:
        _print_judge_response_text(records, args.show_judge_response_limit)
    tasks = _build_tasks(
        records,
        limit=args.limit,
        seed=args.seed,
        debug_filter=args.debug_filter,
        debug_filter_limit=args.debug_filter_limit,
    )
    if not tasks:
        raise SystemExit("No runnable tasks after filtering input records")
    model_configs = _parse_model_configs(args.judge_models)
    num_models = len(model_configs)
    input_path = Path(args.input)

    base_judge_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "requests_per_second": args.requests_per_second,
        "request_timeout": args.request_timeout,
        "api_key_env": args.api_key_env,
        "base_url": args.base_url,
    }

    progress_factory = _tqdm if _tqdm is not None else _SilentTqdm
    aggregated_payload: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input": str(input_path.resolve()),
        "num_samples": len(tasks),
        "models": {},
    }

    try:
        if args.debug:
            debug_config = model_configs[0]
            if num_models > 1:
                print(
                    json.dumps(
                        {
                            "debug": "multiple_models_ignored",
                            "message": "Debug mode only evaluates the first model to print matching samples.",
                            "selected_model": debug_config.alias,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            _evaluate_model_config(
                config=debug_config,
                tasks=tasks,
                args=args,
                input_path=input_path,
                base_judge_params=base_judge_params,
                progress_factory=progress_factory,
                base_temperature=args.temperature,
                debug_high_strength_mode=True,
            )
            return

        with ThreadPoolExecutor(max_workers=len(model_configs) or 1) as model_executor:
            future_map = {
                model_executor.submit(
                    _evaluate_model_config,
                    config=config,
                    tasks=tasks,
                    args=args,
                    input_path=input_path,
                    base_judge_params=base_judge_params,
                    progress_factory=progress_factory,
                    base_temperature=args.temperature,
                    debug_high_strength_mode=False,
                ): config
                for config in model_configs
            }
            for future in as_completed(future_map):
                alias, aggregated_entry, print_payload, failed_examples, errors = future.result()
                aggregated_payload["models"][alias] = aggregated_entry
                print(json.dumps(print_payload, ensure_ascii=False, indent=2))
                if failed_examples:
                    print(
                        json.dumps(
                            {
                                "model": alias,
                                "failed_example_peek": failed_examples,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                if errors:
                    print(
                        f"[INFO] Model {alias} completed with {errors} failed LLM calls",
                        file=sys.stderr,
                    )
    except DebugSampleComplete:
        return

    if args.metrics_json:
        _write_json(args.metrics_json, aggregated_payload)

    if num_models > 1:
        print(json.dumps(aggregated_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
