#!/usr/bin/env python3
"""Standalone adversarial checklist evaluator.

Reads the JSONL emitted by ``generate_adversarial_sets.py`` in which each record already
contains the preferred answer and its adversarial counterpart. For every row, this script
builds a checklist-judge task (preferred vs adversarial), invokes the exact prompt used
by ``arena_cklist.py``, and writes the judge outputs into a CSV—without staging any
per-model JSONL files or shelling out to another script.

Retained capabilities from ``arena_cklist.py``:
- Full checklist prompt and response validation (JSON completion suffix, error retries,
  filtered-response detection).
- Thread-local OpenAI clients with rate limiting, concurrent workers, and resume support.
- Ability to skip tasks already present in existing CSVs (via ``--resume`` or
  ``--exclude-csv``).

Removed capabilities:
- No train/valid/test split files.
- No per-model JSONL work directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from openai import OpenAI
from tqdm import tqdm

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/")
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_PREFERRED_MODEL_NAME = "preferred_answer"
DEFAULT_ADVERSARIAL_MODEL_NAME = "adversarial_answer"
OUTPUT_FIELDNAMES = [
    "question_id",
    "query",
    "model_a",
    "model_b",
    "judge_model",
    "answer_a",
    "answer_b",
    "created_at",
    "response_text",
    "response_json",
    "prompt_input",
    "prompt_preview",
    "dataset_split",
    "violation_rationale",
    "target_bias_type",
    "preservation_notes",
    "attack_strength",
    "swap_id",
]

QUERY_FIELDS = ["new_query", "prompt", "question", "query", "generated_query"]
QUESTION_FALLBACK_FIELDS = ["resume_key", "question_id", "sample_id", "id", "mid", "idx"]
ATTACK_STRENGTH_FIELDS = ["attack_strength", "desired_attack_strength", "attackStrength"]


PROMPT_TEMPLATE = """
You are a logic analysis expert.

Your task: Using the question QUERY and two answer texts (A and B), create a **logic evaluation checklist** that helps annotators compare the answers across eight logical dimensions.

----

## Three-Level Logical Framework
This evaluation follows classical logic and argumentation analysis. Review the response on three layers:
1. **Macro level**: overall direction, task alignment, and global consistency.
2. **Structural level**: completeness of the argumentative chain (claims, evidence, reasoning, qualifiers, consistency).
3. **Expositional level**: information flow from given context to new content, ensuring natural, easy-to-follow transitions.

----

## Eight Logical Dimensions and Focus Points

1. **Task alignment & claim clarity**
    - **Definition**: Does the answer address the core task in prominent locations (opening or closing), present a focused central claim, and stay on topic without digressions?
    - **Focus question**: Which side responds more clearly to the query, states a stronger central position, and maintains thematic discipline?

2. **Global coherence**
    - **Definition**: Evaluates whether the overall structure is organized, follows the task-appropriate sequence (for example, background -> issue -> analysis -> conclusion), and sustains a stable main thread.
    - **Focus question**: Which side shows a more orderly global structure, a task-aligned development sequence, and conclusions that echo the earlier discussion?

3. **Local coherence**
    - **Definition**: Judges whether adjacent sentences or paragraphs connect smoothly, using reasonable transitions between ideas.
    - **Focus question**: Which side offers smoother paragraph transitions and clearer logical bridges between consecutive points?

4. **Evidence sufficiency & relevance**
    - **Definition**: Reviews whether key claims are supported with concrete, on-topic evidence (data, cases, research) and whether the evidence links clearly to conclusions.
    - **Focus question**: Which side provides more specific, relevant evidence and explains how it supports the main conclusions?

5. **Warrants & causal reasoning**
    - **Definition**: Assesses how clearly the response explains why the evidence supports the conclusion, whether the reasoning chain is complete, and whether causal direction is sensible.
    - **Focus question**: Which side presents a more complete causal chain, with clearer explanations of why the evidence leads to the conclusion?

6. **Qualifiers & counterpoints**
    - **Definition**: Looks for acknowledgement of uncertainties, counterexamples, or boundary conditions, and whether qualifiers or concessions keep the argument balanced.
    - **Focus question**: Which side better identifies counterexamples, states conditions, or introduces qualifiers that maintain argumentative balance?

7. **Internal consistency**
    - **Definition**: Checks whether internal data, definitions, and positions stay consistent, without contradictions or drifting interpretations.
    - **Focus question**: Which side maintains more consistent claims, definitions, and stances throughout the answer?

8. **Concept introduction & logical transition**
    - **Definition**: Examines the naturalness of information flow from known context to new content. Evaluate two levels: (a) openings that set up background before introducing the main claim; (b) in-text introductions of new concepts with adequate explanation.
    - **Focus question**: Prioritize which side better prepares the background and introduces the topic at the beginning. If both openings work, compare which side introduces new concepts inside the body with smoother transitions and clearer explanations.

----

## Input Format

[INPUT]
- QUERY
<<<QUERY>>>

- TEXT A
<<<TEXT_A>>>

- TEXT B
<<<TEXT_B>>>

----

## Output Requirements
- Output **only a JSON array with length = 8**; each object corresponds to one of the dimensions above and must appear in the same order.
- Every object must contain the following five fields:
  - "aspect": one of the eight dimension names (use exact names).
  - "question": a concrete comparison question tailored to the current QUERY and answers A/B. It must:
     1. Identify the specific claim or topic under comparison.
     2. Spell out the logical focus (for example, specificity of evidence, completeness of causal chain, quality of transitions).
     3. Be self-contained so an annotator can understand it without rereading the original text.
  - "good_example": an example of what strong performance looks like for this dimension under the stated question.
  - "bad_example": an example of what weak performance looks like for the same question.
  - "span_hint": linguistic signals or logical cues that raters should watch for in the current answers.

----

## Generation Guidelines
- Tailor every checklist item to this specific question and answer pair; avoid empty or templated phrasing.
- Keep "question" precise, focused, and comparative; do not settle for vague prompts like "Which side is more reasonable?"
- For dimension 5 ("Warrants & causal reasoning"), ensure the question explicitly names the causal claim, describes the reasoning path, and states which side explains it more clearly. Do not use arrow notation such as "X->Y->Z"; rewrite it in natural language.
- Make sure "good_example" and "bad_example" address the exact same logical checkpoint described in "question".
- Evaluate logical structure and argumentative quality only; ignore factual accuracy and stylistic polish.
- All content in the checklist must be written in clear, professional English.
- Do not include scores or overall summaries in the checklist.

----

## Output Example (style only)
[
  {
     "aspect": "Evidence sufficiency & relevance",
     "question": "When discussing the claim that carbon taxes reduce emissions, which side provides more concrete evidence (national statistics, case studies, mechanism explanations) and explains how that evidence supports the conclusion?",
     "good_example": "Cites specific national data or research findings and links the figures directly to lower emissions caused by the tax.",
     "bad_example": "Only asserts that \"carbon taxes work\" without any data or explanation.",
     "span_hint": "Look for phrases such as \"research shows\", \"case evidence\", \"data indicate\", followed by a link to the conclusion."
  },
  ...
]

----

"""
# Load adversarial tasks
# ---------------------------------------------------------------------------


@dataclass
class JudgmentTask:
    question_id: str
    query: str
    model_a: str
    text_a: str
    model_b: str
    text_b: str
    violation_rationale: str = ""
    target_bias_type: str = ""
    preservation_notes: str = ""
    attack_strength: str = ""
    swap_id: str = ""


def _expand_tasks_with_swaps(tasks: Sequence[JudgmentTask]) -> List[JudgmentTask]:
    expanded: List[JudgmentTask] = []
    for task in tasks:
        base_swap_id = f"{task.question_id}-orig"
        swapped_swap_id = f"{task.question_id}-swap"
        expanded.append(replace(task, swap_id=base_swap_id))
        expanded.append(
            JudgmentTask(
                question_id=task.question_id,
                query=task.query,
                model_a=task.model_b,
                text_a=task.text_b,
                model_b=task.model_a,
                text_b=task.text_a,
                violation_rationale=task.violation_rationale,
                target_bias_type=task.target_bias_type,
                preservation_notes=task.preservation_notes,
                attack_strength=task.attack_strength,
                swap_id=swapped_swap_id,
            )
        )
    return expanded


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield idx, data


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _canonical_label(value: Any) -> str:
    text = _normalize_text(value).upper()
    if text in {"A", "B"}:
        return text
    if text in {"ANSWER_A", "PREFERRED_A"}:
        return "A"
    if text in {"ANSWER_B", "PREFERRED_B"}:
        return "B"
    return ""


def _resolve_attack_strength(record: Dict[str, Any]) -> str:
    for key in ATTACK_STRENGTH_FIELDS:
        raw = record.get(key)
        if raw not in (None, ""):
            return _normalize_text(raw).lower()
    return ""


def _parse_checklist_index(record: Dict[str, Any]) -> Optional[int]:
    value = record.get("checklist_index")
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = _normalize_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _extract_query(record: Dict[str, Any]) -> Optional[str]:
    for key in QUERY_FIELDS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _build_question_id(record: Dict[str, Any], fallback_index: int) -> Optional[str]:
    components: List[str] = []
    for key in QUESTION_FALLBACK_FIELDS:
        raw = record.get(key)
        if raw not in (None, ""):
            text = _normalize_text(raw)
            if text:
                if key == "resume_key":
                    return text
                components.append(text)
                break
    if not components:
        components.append(f"row-{fallback_index}")
    ck_idx = _parse_checklist_index(record)
    if ck_idx is not None:
        components.append(f"ck{ck_idx}")
    strength = _resolve_attack_strength(record)
    if strength:
        components.append(strength)
    return "::".join(components) if components else None


def _choose_preferred_answer(record: Dict[str, Any]) -> Optional[str]:
    label = _canonical_label(record.get("target_label") or record.get("original_label"))
    answer_a = record.get("answer_a") or record.get("text_a")
    answer_b = record.get("answer_b") or record.get("text_b")
    candidate_fields = ["preferred_answer", "favored_answer", "original_answer"]
    if label == "A" and answer_a:
        return _normalize_text(answer_a)
    if label == "B" and answer_b:
        return _normalize_text(answer_b)
    for field in candidate_fields:
        candidate = record.get(field)
        if candidate:
            return _normalize_text(candidate)
    return None


def load_tasks(
    adversarial_path: Path,
    *,
    model_a_name: str,
    model_b_name: str,
    limit: Optional[int],
    seed: Optional[int],
) -> Tuple[List[JudgmentTask], Counter]:
    stats: Counter = Counter()
    collected: List[JudgmentTask] = []

    for row_index, record in iter_jsonl(adversarial_path):
        stats["total"] += 1
        query = _extract_query(record)
        if not query:
            stats["missing_query"] += 1
            continue
        adversarial_answer = _normalize_text(record.get("adversarial_answer"))
        if not adversarial_answer:
            stats["missing_adversarial"] += 1
            continue
        preferred_answer = _choose_preferred_answer(record)
        if not preferred_answer:
            stats["missing_preferred"] += 1
            continue
        question_id = _build_question_id(record, row_index)
        if not question_id:
            stats["missing_question_id"] += 1
            continue
        violation_rationale = _normalize_text(record.get("violation_rationale"))
        target_bias_type = _normalize_text(
            record.get("target_bias_type")
            or record.get("bias_type")
        )
        preservation_notes = _normalize_text(record.get("preservation_notes"))
        attack_strength = _resolve_attack_strength(record)
        collected.append(
            JudgmentTask(
                question_id=question_id,
                query=query,
                model_a=model_a_name,
                text_a=preferred_answer,
                model_b=model_b_name,
                text_b=adversarial_answer,
                violation_rationale=violation_rationale,
                target_bias_type=target_bias_type,
                preservation_notes=preservation_notes,
                attack_strength=attack_strength,
            )
        )

    unique_tasks: List[JudgmentTask] = []
    seen_qids: Set[str] = set()
    duplicates = 0
    for task in collected:
        if task.question_id in seen_qids:
            duplicates += 1
            continue
        seen_qids.add(task.question_id)
        unique_tasks.append(task)
    if duplicates:
        stats["duplicate_question_id"] = duplicates

    if seed is not None:
        random.Random(seed).shuffle(unique_tasks)
    if limit is not None:
        unique_tasks = unique_tasks[:limit]
        stats["capped"] = limit

    stats["selected"] = len(unique_tasks)
    return unique_tasks, stats


# ---------------------------------------------------------------------------
# Judge client & execution
# ---------------------------------------------------------------------------


_thread_local = threading.local()


class MissingAPIKeyError(RuntimeError):
    pass


class FilteredResponseError(RuntimeError):
    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code or "filtered"


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
class JudgmentResult:
    task: JudgmentTask
    judge_model: str
    response_text: str
    response_json: Optional[Dict[str, Any]]
    created_at: str
    prompt_input: str


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
    ) -> None:
        if not OPENAI_API_KEY:
            raise MissingAPIKeyError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout
        self.rate_limiter = _get_rate_limiter(requests_per_second)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _is_filtered_error(exc: Exception) -> bool:
        message = str(exc)
        if "The response was filtered" in message:
            return True
        if "code': '1209'" in message or '"code": "1209"' in message:
            return True
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                data = response.json()
                code = data.get("error", {}).get("code")
                if code == "1209":
                    return True
            except Exception:
                pass
        return False

    def score(self, task: JudgmentTask) -> JudgmentResult:
        _, user_content = build_prompt(task)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a meticulous evaluation assistant. Follow the provided instructions exactly, "
                    "carry out every step, and respond only with a single JSON object that matches the requested schema."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                call_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "max_completion_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": messages,
                    "stream": False,
                }
                if self.request_timeout is not None:
                    call_kwargs["timeout"] = self.request_timeout

                completion = self.client.chat.completions.create(**call_kwargs)
                if completion and completion.choices:
                    content = completion.choices[0].message.content or ""
                else:
                    content = ""
                parsed = self._extract_json(content)
                return JudgmentResult(
                    task=task,
                    judge_model=self.model,
                    response_text=content,
                    response_json=parsed,
                    created_at=datetime.utcnow().isoformat() + "Z",
                    prompt_input=user_content,
                )
            except Exception as exc:
                if self._is_filtered_error(exc):
                    raise FilteredResponseError("Judge response filtered", code="1209") from exc
                last_error = exc
                print(f"JudgeClient attempt {attempt} failed: {exc}")
                time.sleep(min(3 * attempt, 10))
        raise RuntimeError(f"Judge request failed after {self.max_attempts} attempts: {last_error}")


def get_thread_judge_client(params: Dict[str, Any]) -> JudgeClient:
    key = (
        params["model"],
        params["temperature"],
        params["max_tokens"],
        params["max_attempts"],
        params["requests_per_second"],
        params["request_timeout"],
    )
    cached = getattr(_thread_local, "judge_client_info", None)
    if cached is None or cached[0] != key:
        client = JudgeClient(
            model=params["model"],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            max_attempts=params["max_attempts"],
            requests_per_second=params["requests_per_second"],
            request_timeout=params["request_timeout"],
        )
        _thread_local.judge_client_info = (key, client)
        return client
    return cached[1]


def build_prompt(task: JudgmentTask) -> Tuple[str, str]:
    base_prompt = (
        PROMPT_TEMPLATE.replace("<<TEXT_A>>", task.text_a)
        .replace("<<TEXT_B>>", task.text_b)
        .replace("<<<QUERY>>>", task.query)
    )
    return base_prompt, base_prompt


def score_task_with_client(task: JudgmentTask, params: Dict[str, Any]) -> Dict[str, Any]:
    client = get_thread_judge_client(params)
    result = client.score(task)
    response_json_str = ""
    if result.response_json is not None:
        response_json_str = json.dumps(result.response_json, ensure_ascii=False)
    return {
        "question_id": task.question_id,
        "query": task.query,
        "model_a": task.model_a,
        "model_b": task.model_b,
        "judge_model": params["model"],
        "answer_a": task.text_a,
        "answer_b": task.text_b,
        "created_at": result.created_at,
        "response_text": result.response_text,
        "response_json": response_json_str,
        "prompt_input": result.prompt_input,
        "prompt_preview": "",
        "dataset_split": "",
        "violation_rationale": task.violation_rationale,
        "target_bias_type": task.target_bias_type,
        "preservation_notes": task.preservation_notes,
        "attack_strength": task.attack_strength,
        "swap_id": task.swap_id,
    }


def safe_score_task(task: JudgmentTask, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return score_task_with_client(task, params)
    except FilteredResponseError as exc:
        error_payload = {"error": {"code": exc.code, "message": str(exc)}}
        print(f"Skipping question {task.question_id}: judge output was filtered ({exc.code})")
        return {
            "question_id": task.question_id,
            "query": task.query,
            "model_a": task.model_a,
            "model_b": task.model_b,
            "judge_model": params["model"],
            "answer_a": task.text_a,
            "answer_b": task.text_b,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "response_text": "",
            "response_json": json.dumps(error_payload, ensure_ascii=False),
            "prompt_input": build_prompt(task)[0],
            "prompt_preview": "",
            "dataset_split": "",
            "violation_rationale": task.violation_rationale,
            "target_bias_type": task.target_bias_type,
            "preservation_notes": task.preservation_notes,
            "attack_strength": task.attack_strength,
            "swap_id": task.swap_id,
        }
    except RuntimeError as exc:
        error_payload = {"error": {"code": "judge_runtime_error", "message": str(exc)}}
        print(
            f"Skipping question {task.question_id}: exhausted {params.get('max_attempts', 0)} consecutive attempts"
        )
        return {
            "question_id": task.question_id,
            "query": task.query,
            "model_a": task.model_a,
            "model_b": task.model_b,
            "judge_model": params["model"],
            "answer_a": task.text_a,
            "answer_b": task.text_b,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "response_text": "",
            "response_json": json.dumps(error_payload, ensure_ascii=False),
            "prompt_input": build_prompt(task)[0],
            "prompt_preview": "",
            "dataset_split": "",
            "violation_rationale": task.violation_rationale,
            "target_bias_type": task.target_bias_type,
            "preservation_notes": task.preservation_notes,
            "attack_strength": task.attack_strength,
            "swap_id": task.swap_id,
        }


# ---------------------------------------------------------------------------
# Exclusion helpers
# ---------------------------------------------------------------------------


def load_exclusion_pairs(paths: Sequence[str]) -> Set[Tuple[str, str, str]]:
    exclusion_pairs: Set[Tuple[str, str, str]] = set()
    for path in paths:
        if not path:
            continue
        rows = _load_output_rows(path)
        for row in rows:
            key = _row_key(row)
            if key is None:
                continue
            qid, model_a, model_b = key
            exclusion_pairs.add(_canonical_pair_key(qid, model_a, model_b))
    return exclusion_pairs


def load_existing_results(path: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    processed: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                qid = row.get("question_id")
                model_a = row.get("model_a")
                model_b = row.get("model_b")
                if not qid or not model_a or not model_b:
                    continue
                key = (qid, model_a, model_b)
                processed[key] = row
    except Exception as exc:
        print(f"Warning: Failed to read existing CSV results {path}: {exc}")
    return processed


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge preferred vs adversarial answers directly")
    parser.add_argument("--adversarial-jsonl", required=True, help="Path to adversarial_outputs.jsonl")
    parser.add_argument("--out-jsonl", required=True, help="CSV path for judge outputs")
    parser.add_argument("--preferred-model-name", default=DEFAULT_PREFERRED_MODEL_NAME)
    parser.add_argument("--adversarial-model-name", default=DEFAULT_ADVERSARIAL_MODEL_NAME)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=35000)
    parser.add_argument("--num-workers", type=int, default=128)
    parser.add_argument("--requests-per-second", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42, help="Used for shuffling before --limit")
    parser.add_argument("--limit", type=int, help="Optional cap on number of tasks")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already completed in --out-jsonl")
    parser.add_argument("--exclude-csv", nargs="*", default=[], help="Existing outputs to skip as well")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only materialize prompts into CSV without calling the judge",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Stop at the first incomplete response by printing the model output and prompt",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    adversarial_path = Path(args.adversarial_jsonl)
    if not adversarial_path.exists():
        raise SystemExit(f"Adversarial JSONL not found: {adversarial_path}")
    if args.preferred_model_name == args.adversarial_model_name:
        raise SystemExit("preferred-model-name and adversarial-model-name must differ")

    tasks, stats = load_tasks(
        adversarial_path,
        model_a_name=args.preferred_model_name,
        model_b_name=args.adversarial_model_name,
        limit=args.limit,
        seed=args.seed,
    )
    if not tasks:
        raise SystemExit(f"No eligible adversarial rows (stats={dict(stats)}).")

    base_task_count = len(tasks)
    tasks = _expand_tasks_with_swaps(tasks)

    print(
        "Selected {} adversarial rows (stats: {}).".format(
            base_task_count,
            {
                "total": stats.get("total", 0),
                "missing_query": stats.get("missing_query", 0),
                "missing_preferred": stats.get("missing_preferred", 0),
                "missing_adversarial": stats.get("missing_adversarial", 0),
                "missing_question_id": stats.get("missing_question_id", 0),
                "duplicate_question_id": stats.get("duplicate_question_id", 0),
                "capped": stats.get("capped", 0),
            },
        )
    )
    print(f"Expanded to {len(tasks)} judge tasks after applying A/B swaps.")

    existing_rows = _load_output_rows(args.out_jsonl)
    if existing_rows:
        complete_rows, incomplete_keys = _split_rows_by_completeness(existing_rows)
        if incomplete_keys:
            print(
                f"Detected {len(incomplete_keys)} incomplete response(s) in existing output; rewriting file before rerun."
            )
            _rewrite_output_rows(args.out_jsonl, complete_rows)

    existing: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if args.resume:
        existing = load_existing_results(args.out_jsonl)

    exclusion_pairs = load_exclusion_pairs(args.exclude_csv)
    tasks_to_run: List[Tuple[Tuple[str, str, str], JudgmentTask]] = []
    skipped = 0
    task_map: Dict[Tuple[str, str, str], JudgmentTask] = {}
    for task in tasks:
        key = (task.question_id, task.model_a, task.model_b)
        task_map[key] = task
        if args.resume and key in existing:
            existing_response = existing[key].get("response_text") or ""
            if existing_response and _is_response_complete(existing_response):
                skipped += 1
                continue
        if exclusion_pairs and _canonical_pair_key(*key) in exclusion_pairs:
            skipped += 1
            continue
        tasks_to_run.append((key, task))
    if skipped:
        print(f"Skipped {skipped} existing evaluations (resume/exclude mode).")
    if not tasks_to_run:
        print("No tasks to run after filtering.")
        return

    if args.dry_run:
        rows: List[Dict[str, Any]] = []
        for _, task in tasks_to_run:
            prompt, _ = build_prompt(task)
            prompt_preview = prompt if len(prompt) <= 4000 else prompt[:4000] + "\n... [truncated]"
            rows.append(
                {
                    "question_id": task.question_id,
                    "query": task.query,
                    "model_a": task.model_a,
                    "model_b": task.model_b,
                    "judge_model": args.judge_model,
                    "answer_a": task.text_a,
                    "answer_b": task.text_b,
                    "created_at": "",
                    "response_text": "",
                    "response_json": "",
                    "prompt_input": prompt,
                    "prompt_preview": prompt_preview,
                    "dataset_split": "",
                    "violation_rationale": task.violation_rationale,
                    "target_bias_type": task.target_bias_type,
                    "preservation_notes": task.preservation_notes,
                    "attack_strength": task.attack_strength,
                    "swap_id": task.swap_id,
                }
            )
        write_rows_csv(args.out_jsonl, rows)
        return

    judge_params = {
        "model": args.judge_model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "requests_per_second": args.requests_per_second,
        "request_timeout": args.request_timeout,
    }

    new_rows: List[Dict[str, Any]] = []
    pending_keys: Set[Tuple[str, str, str]] = set(key for key, _ in tasks_to_run)
    attempt_counts: Dict[Tuple[str, str, str], int] = {key: 0 for key in pending_keys}
    exception_retry_limit = max(1, args.max_attempts)

    debug_triggered = False

    progress_bar = tqdm(total=len(tasks_to_run), desc="Judging tasks", unit="task")

    def mark_task_finished(key: Tuple[str, str, str]) -> None:
        if key in pending_keys:
            pending_keys.remove(key)
            progress_bar.update(1)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
            future_map: Dict[Any, Tuple[str, str, str]] = {}
            for key, task in tasks_to_run:
                future = executor.submit(safe_score_task, task, judge_params)
                future_map[future] = key

            for future in as_completed(future_map):
                key = future_map[future]
                task_label = f"{key[0]}:{key[1]}|{key[2]}"
                try:
                    row = future.result()
                except Exception as exc:
                    attempt_counts[key] += 1
                    if attempt_counts[key] >= exception_retry_limit:
                        print(
                            f"[WARN] Persistent exception for {task_label}: {exc}. Recording failure and moving on."
                        )
                        error_payload = {
                            "error": {"code": "judge_unhandled_exception", "message": str(exc)}
                        }
                        new_rows.append(
                            {
                                "question_id": key[0],
                                "query": task_map[key].query,
                                "model_a": key[1],
                                "model_b": key[2],
                                "judge_model": args.judge_model,
                                "answer_a": task_map[key].text_a,
                                "answer_b": task_map[key].text_b,
                                "created_at": datetime.utcnow().isoformat() + "Z",
                                "response_text": "",
                                "response_json": json.dumps(error_payload, ensure_ascii=False),
                                "prompt_input": build_prompt(task_map[key])[0],
                                "prompt_preview": "",
                                "dataset_split": "",
                                "violation_rationale": task_map[key].violation_rationale,
                                "target_bias_type": task_map[key].target_bias_type,
                                "preservation_notes": task_map[key].preservation_notes,
                                "attack_strength": task_map[key].attack_strength,
                                "swap_id": task_map[key].swap_id,
                            }
                        )
                        mark_task_finished(key)
                        attempt_counts[key] = 0
                    else:
                        print(
                            f"[WARN] Exception for {task_label}: {exc}. Retrying later (attempt {attempt_counts[key]})."
                        )
                        future = executor.submit(safe_score_task, task_map[key], judge_params)
                        future_map[future] = key
                    continue

                response_text = row.get("response_text", "") or ""
                if _is_response_complete(response_text):
                    new_rows.append(row)
                    mark_task_finished(key)
                    attempt_counts[key] = 0
                else:
                    if response_text:
                        attempt_counts[key] += 1
                        warn_message = (
                            f"[WARN] Incomplete response for {task_label}; retrying (attempt {attempt_counts[key]})."
                        )
                        print(warn_message)
                        if args.debug:
                            prompt_input = row.get("prompt_input", "") or ""
                            print("[DEBUG] Response text:\n" + response_text)
                            print("[DEBUG] Prompt input:\n" + prompt_input)
                            for pending_future in future_map:
                                if not pending_future.done():
                                    pending_future.cancel()
                            debug_triggered = True
                            break
                        future = executor.submit(safe_score_task, task_map[key], judge_params)
                        future_map[future] = key
                    else:
                        new_rows.append(row)
                        mark_task_finished(key)
    finally:
        progress_bar.close()

    write_rows_csv(args.out_jsonl, new_rows)
    if args.debug and debug_triggered:
        print("Debug mode triggered by incomplete response; exiting early after dumping context.")
        return
    if os.path.exists(args.out_jsonl):
        remaining_incomplete = _find_incomplete_output_keys(args.out_jsonl)
        if remaining_incomplete:
            raise RuntimeError(
                f"Detected {len(remaining_incomplete)} incomplete response(s) after retries."
            )


if __name__ == "__main__":
    main()
