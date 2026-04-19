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
    "checklist_aspect",
    "preservation_notes",
    "attack_strength",
    "swap_id",
]

QUERY_FIELDS = ["new_query", "prompt", "question", "query", "generated_query"]
QUESTION_FALLBACK_FIELDS = ["resume_key", "question_id", "sample_id", "id", "mid", "idx"]
ATTACK_STRENGTH_FIELDS = ["attack_strength", "desired_attack_strength", "attackStrength"]


PROMPT_TEMPLATE = """
You are a logic analysis expert.

Your task: Given the question QUERY and two answer texts (A and B), produce a **logic evaluation checklist** that helps annotators compare the answers across eight logical dimensions.

---

## Three-Level Logic Framework
This evaluation draws on classical logic and argumentation analysis theory and divides textual logic into three layers:
1. **Macro level**: Examine the overall argumentative direction and thematic focus; determine whether the answer aligns with the task and maintains global coherence.
2. **Structural level**: Assess the completeness of the reasoning chain, including claims, evidence, warrants, qualifiers, and internal consistency.
3. **Expositional level**: Using information-flow theory (Given -> New), judge whether the progression from familiar to new information is natural, smooth, and easy to follow.

---

## Eight Logical Dimensions and Question Focus

1. **Task alignment & claim clarity**  
    - **Definition**: Judge whether the answer (particularly the opening or conclusion) clearly addresses the core task in QUERY, presents a focused central claim, and maintains topical alignment throughout.  
    - **Question focus**: Which side more clearly addresses the task up front or in the conclusion, states a central stance, and keeps the discussion on-topic?

2. **Global coherence**  
    - **Definition**: Evaluate whether the overall structure is logical from start to finish and follows an order appropriate for QUERY (e.g., background -> issue -> analysis -> conclusion for analytical tasks, or comparison -> evaluation -> conclusion for comparison tasks). Focus on whether the essay sustains a stable argumentative arc.  
    - **Question focus**: Which side has a more organized global structure, follows a task-appropriate development sequence, and lands a conclusion that ties back to the body?

3. **Local coherence**  
    - **Definition**: Inspect the logical relationships between adjacent paragraphs or sentences—whether transitions are smooth and arguments connect naturally without leaps.  
    - **Question focus**: Which side provides smoother paragraph transitions and more coherent local reasoning links?

4. **Evidence sufficiency & relevance**  
    - **Definition**: Determine whether key claims are supported by sufficient, on-topic evidence (data, experiments, facts, literature, case studies, etc.) and whether that evidence is logically connected to the conclusions.  
    - **Question focus**: Which side offers more specific, relevant evidence that directly supports the main conclusions?

5. **Warrants & causal reasoning**  
    - **Definition**: Assess whether the answer explains *why* the evidence supports the conclusions, whether the reasoning chain is complete, and whether causal directions are valid without leaps.  
    - **Question focus**: Which side provides a more complete reasoning chain with clearer causal explanations and more reasonable logical transitions?

6. **Qualifiers & counterpoints**  
    - **Definition**: Check whether the answer acknowledges counterexamples, uncertainties, or exceptions, and whether it uses qualifiers, concessions, or balanced phrasing to avoid overstatements.  
    - **Question focus**: Which side better identifies counterpoints or boundary conditions and uses them to keep the argument balanced?

7. **Internal consistency**  
    - **Definition**: Inspect whether data, definitions, and positions remain consistent throughout the answer, avoiding contradictions in numbers, terminology, or stance.  
    - **Question focus**: Which side stays more internally consistent and self-coherent?

8. **Concept introduction & logical transition**  
    - **Definition**: Evaluate the naturalness of information development, especially the transition from known to new concepts.  
      Assessment proceeds in two stages:  
      1. **Opening level**: Does the introduction supply the necessary background, context, or problem framing before naturally leading to the main topic or claim?  
          - Jumping directly into abstract discussion or unexplained terminology counts as a logical gap.  
      2. **Within-body level**: When introducing new concepts, models, or terms in the body, does the answer provide concise explanations or context so readers can grasp their role?  
    - **Question focus**:  
      First compare which side eases the reader in with background and transitions smoothly into the topic. If both introductions are solid, then compare which body provides smoother transitions and clearer explanations when introducing new concepts.

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

## Output Requirements
- Output **only a JSON array of length 8**, where each object corresponds to one of the eight dimensions in order.
- Each object must contain the following five fields:  
  - `"aspect"`: One of the eight dimension names (exact naming).  
  - `"question"`: A dimension-specific evaluation question tailored to the current QUERY and answers A/B.  
     - The question must reference the current context and highlight a concrete logical phenomenon; avoid abstract prompts such as "Is it reasonable?" or "Is it sufficient?"  
     - The question must enable the annotator to decide directly which side performs better.  
  - `"good_example"`: A positive example that demonstrates what a strong answer would do for this question.  
  - `"bad_example"`: A negative example that demonstrates what a weak answer would do for the same question.  
     - Both examples must tie to the same logical focus described in the question.  
  - `"span_hint"`: Natural-language cues that direct annotators to the relevant signals or logic patterns in the answers.

---

## Generation Guidelines
- The checklist must be **fully customized to the specific task**, not generic or templated.  
- Each `question` must satisfy three properties: semantic self-sufficiency, a precise logical focus, and explicit comparison targets.  
  Specifically:  
  1. Clearly identify the claim or topic under comparison (e.g., "carbon taxes reduce emissions" rather than "key claim").  
  2. Spell out the logical focus (e.g., "specificity and relevance of evidence," "completeness of the causal chain," "smoothness of paragraph transitions") rather than vague adjectives.  
  3. Use natural language so annotators can understand the question without re-reading the entire context.  
      - Avoid: "Which side provides more concrete support for the key claim?"  
      - Prefer: "Between A and B, which side offers more specific evidence (such as emissions data, case studies, or mechanism explanations) supporting the claim that carbon taxes reduce emissions, and explains how the evidence backs the conclusion?"
- Do not rely on bracketed terminology notes (e.g., "MVO = variance risk") unless that notation appears and is defined in the original answers.
- If the answers contain multiple claims, choose the most central or frequently emphasized claim as the focal point of the question.
- For dimension 5 (**Warrants & causal reasoning**) specifically:  
  - The `question` must identify the exact reasoning chain or causal direction discussed by A/B, rather than asking generically whether the logic is sound.  
    - Do not express causal chains using symbols or arrows (such as "X->Y->Z"); rewrite them in natural language ("X leads to Y, which in turn affects Z").  
  - The question must include: 1) the causal object (the claim being explained), 2) the causal path (why or how it holds), and 3) the comparison focus (which side gives a clearer account).  
  - Example:  
     - Weak: "Which side better explains why the indicators support accurate evaluation?" (too abstract)  
     - Improved: "Between A and B, which side more clearly explains why policy innovation, rising market demand, and technological progress together drive sustainable development in real estate?"
- `good_example` and `bad_example` must align with the same logical focus and illustrate the positive vs. negative manifestations of that focal point.  
- Evaluate **logical structure and argumentative quality only**; ignore factual accuracy and writing style.  
- All outputs must be written in fluent, concise, and readable English.  
- Do not add scores or overall summaries.

---

## Output Style Example
[
  {
     "aspect": "Evidence sufficiency & relevance",
     "question": "Between A and B, which side provides more specific evidence (such as emissions data, national case studies, or mechanism descriptions) to support the claim that carbon taxes reduce emissions, and explains how the evidence underpins the conclusion?",
     "good_example": "Cites concrete national data or research findings and explains how carbon taxes push firms to curb emissions, thereby supporting the claim.",
     "bad_example": "Only asserts that carbon taxes work without offering data or logical explanation.",
     "span_hint": "Look for phrases such as 'data show,' 'studies find,' or 'case evidence' that directly connect evidence to the conclusion."
  },
  ...
]

---

Strictly follow the instructions above and output only a JSON array (length = 8) that matches the required keys and structure, with all descriptions in English.
"""


# ---------------------------------------------------------------------------
# CSV helpers (based on arena_cklist)
# ---------------------------------------------------------------------------


def _configure_csv_field_size_limit() -> None:
    env_limit = os.environ.get("CSV_FIELD_SIZE_LIMIT")
    default_limit = 64 * 1024 * 1024
    limit = default_limit
    if env_limit:
        try:
            limit = max(int(env_limit), default_limit)
        except ValueError:
            print(f"Warning: Invalid CSV_FIELD_SIZE_LIMIT '{env_limit}', using default {default_limit}")
            limit = default_limit
    try:
        csv.field_size_limit(limit)
    except OverflowError:
        csv.field_size_limit(sys.maxsize)


_configure_csv_field_size_limit()


def _is_response_complete(response_text: str) -> bool:
    if not response_text:
        return False
    return bool(response_text.strip())


def _load_output_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path) or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]
    except Exception as exc:
        print(f"Warning: Failed to read existing CSV {path}: {exc}")
        return []


def _row_key(row: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    question_id = (row.get("question_id") or "").strip()
    model_a = (row.get("model_a") or "").strip()
    model_b = (row.get("model_b") or "").strip()
    if not question_id or not model_a or not model_b:
        return None
    return question_id, model_a, model_b


def _canonical_pair_key(question_id: str, model_a: str, model_b: str) -> Tuple[str, str, str]:
    if model_a <= model_b:
        return question_id, model_a, model_b
    return question_id, model_b, model_a


def _split_rows_by_completeness(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Set[Tuple[str, str, str]]]:
    complete_rows: List[Dict[str, Any]] = []
    incomplete_keys: Set[Tuple[str, str, str]] = set()
    for row in rows:
        key = _row_key(row)
        if key is None:
            continue
        response_text = row.get("response_text", "") or ""
        if not response_text or _is_response_complete(response_text):
            complete_rows.append(row)
        else:
            incomplete_keys.add(key)
    return complete_rows, incomplete_keys


def _normalize_row(row: Dict[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for field in OUTPUT_FIELDNAMES:
        value = row.get(field)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif value is None:
            value = ""
        else:
            value = str(value)
        normalized[field] = value
    return normalized


def _rewrite_output_rows(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))


def _find_incomplete_output_keys(path: str) -> Set[Tuple[str, str, str]]:
    rows = _load_output_rows(path)
    _, incomplete_keys = _split_rows_by_completeness(rows)
    return incomplete_keys


def write_rows_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    needs_rewrite = False
    existing_rows: List[Dict[str, Any]] = []
    if file_exists:
        with open(path, "r", encoding="utf-8", newline="") as existing_fh:
            reader = csv.DictReader(existing_fh)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != OUTPUT_FIELDNAMES:
                needs_rewrite = True
                existing_rows.extend(reader)
    if needs_rewrite:
        with open(path, "w", encoding="utf-8", newline="") as rewrite_fh:
            writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDNAMES)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(_normalize_row(row))
        file_exists = True
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))


# ---------------------------------------------------------------------------
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
    checklist_aspect: str = ""
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
                checklist_aspect=task.checklist_aspect,
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
        checklist_aspect = _normalize_text(record.get("checklist_aspect"))
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
                checklist_aspect=checklist_aspect,
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
        "checklist_aspect": task.checklist_aspect,
        "preservation_notes": task.preservation_notes,
        "attack_strength": task.attack_strength,
        "swap_id": task.swap_id,
    }


def safe_score_task(task: JudgmentTask, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return score_task_with_client(task, params)
    except FilteredResponseError as exc:
        error_payload = {"error": {"code": exc.code, "message": str(exc)}}
        print(f"Skip question {task.question_id}: judge output was filtered ({exc.code})")
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
            "checklist_aspect": task.checklist_aspect,
            "preservation_notes": task.preservation_notes,
            "attack_strength": task.attack_strength,
            "swap_id": task.swap_id,
        }
    except RuntimeError as exc:
        error_payload = {"error": {"code": "judge_runtime_error", "message": str(exc)}}
        print(f"Skip question {task.question_id}: failed {params.get('max_attempts', 0)} consecutive attempts")
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
            "checklist_aspect": task.checklist_aspect,
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
                    "checklist_aspect": task.checklist_aspect,
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
                                "checklist_aspect": task_map[key].checklist_aspect,
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
