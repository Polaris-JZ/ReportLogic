#!/usr/bin/env python3
"""Construct adversarial answer variants targeting individual checklist items.

This script samples rows from one or more CSVs that follow the same format as `vllm_infer.py`
inputs, then uses an LLM (via the same OpenAI-compatible endpoint employed by
`arena_cklist.py`) to generate adversarial modifications of the preferred answer.
During generation it reports progress with `tqdm` when available, runs requests in parallel across configurable worker threads, and flushes JSONL output incrementally with a configurable batch size (default 10 records) to avoid data loss during long runs.
It also supports a resume mode that reuses the existing output JSONL, skipping any task whose unique key (query + answer_a + answer_b + checklist aspect + attack strength) has already been produced.
For every sampled row, eight adversarial variants are produced—one for each
checklist dimension—where the modified answer intentionally violates the target
logic item while keeping the other dimensions intact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

CHECKLIST_COUNT = 8
CHECKLIST_PREFIX = "cklist_"
DEFAULT_LABEL_COLUMN = "overall_winner"
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_OUTPUT_PATH = "adversarial_outputs.jsonl"
DEFAULT_GENERATOR_MODEL = "gpt-5.1"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 30720
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_NUM_WORKERS = 128
CSV_ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "gb18030", "gbk"]
DEFAULT_ATTACK_STRENGTHS = ("low", "medium", "high")
DEFAULT_LANGUAGE_RETRY_LIMIT = 3
ATTACK_STRENGTH_CHOICES = ("low", "medium", "high")

ANT_OPENAI_API_KEY_V2 = os.environ.get("ANT_OPENAI_API_KEY_V2")
BASE_URL = os.environ.get("ANT_OPENAI_BASE_URL", "http://matrixllm-pool.global.alipay.com/v1/")

WINNER_NORMALIZATION_MAP = {
    "assistant_1": "A",
    "assistant 1": "A",
    "assistant_a": "A",
    "a": "A",
    "a>b": "A",
    "a > b": "A",
    "a_win": "A",
    "a wins": "A",
    "assistant_2": "B",
    "assistant 2": "B",
    "assistant_b": "B",
    "b": "B",
    "b>a": "B",
    "b > a": "B",
    "b_win": "B",
    "b wins": "B",
    "both good": "both_good",
    "both_good": "both_good",
    "both bad": "both_bad",
    "both_bad": "both_bad",
}

VALUE_PAIR_CANDIDATES = [
    ("answer_a", "answer_b"),
    ("text_a", "text_b"),
    ("response_a", "response_b"),
    ("assistant_1", "assistant_2"),
    ("assistant_a", "assistant_b"),
    ("candidate_a", "candidate_b"),
]

OTHER_ASPECT_NAMES = [
    "Task alignment & claim clarity (Task alignment & claim clarity)",
    "Global coherence (Global coherence)",
    "Local coherence (Local coherence)",
    "Evidence sufficiency & relevance (Evidence sufficiency & relevance)",
    "Warrants & causal reasoning (Warrants & causal reasoning)",
    "Qualifiers & counterpoints (Qualifiers & counterpoints)",
    "Internal consistency (Internal consistency)",
    "Concept introduction & logical transition (Concept introduction & logical transition)",
]

ASPECT_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "Task alignment & claim clarity (Task alignment & claim clarity)": {
        "definition": "Focuses on whether the answer responds to the QUERY's core task at a prominent position and presents a clear, focused claim without digressing.",
        "question_focus": "Compares which side more clearly responds to the question at the beginning or end, presents a central position, and maintains thematic consistency.",
    },
    "Global coherence (Global coherence)": {
        "definition": "Examines whether the overall logical architecture is clear, whether it unfolds in the sequence required by the task, and forms a stable main thread.",
        "question_focus": "Compares which side has a more organized structure, follows a development sequence that better aligns with task logic, and has conclusions that echo the preceding text.",
    },
    "Local coherence (Local coherence)": {
        "definition": "Focuses on whether the transitions between adjacent paragraphs or sentences are smooth and whether there are logical breaks or jumps.",
        "question_focus": "Compares which side has more reasonable paragraph connections and more coherent transitions.",
    },
    "Evidence sufficiency & relevance (Evidence sufficiency & relevance)": {
        "definition": "Evaluates whether key claims are supported by sufficient and relevant evidence, and whether evidence and conclusions form clear logic.",
        "question_focus": "Compares which side provides more specific and relevant evidence and explains how it supports the main conclusions.",
    },
    "Warrants & causal reasoning (Warrants & causal reasoning)": {
        "definition": "Checks whether the answer explains why the evidence supports the conclusion, and whether the causal chain is complete and directionally sound.",
        "question_focus": "Compares which side has a more complete reasoning chain, clearer causal relationships, and more reasonable logical transitions.",
    },
    "Qualifiers & counterpoints (Qualifiers & counterpoints)": {
        "definition": "Focuses on whether counterexamples or uncertainties are identified, and whether qualifiers or concessions are used to maintain argumentative balance.",
        "question_focus": "Compares which side better identifies counterexamples, sets prerequisites or qualifiers, and keeps the argument balanced.",
    },
    "Internal consistency (Internal consistency)": {
        "definition": "Evaluates whether internal data, definitions, and positions are consistent throughout the text without obvious contradictions.",
        "question_focus": "Compares which side has more consistent discourse and more self-consistent internal logic.",
    },
    "Concept introduction & logical transition (Concept introduction & logical transition)": {
        "definition": "Focuses on whether the information flow from known context to new content is natural, whether the opening provides suitable background, and whether new concepts are explained when introduced.",
        "question_focus": "Compares which side better lays out the background and provides natural transitions and explanations when introducing new concepts.",
    },
}

DEFAULT_ASPECT_DEFINITION = "This dimension captures how the answer behaves on a specific logical criterion. Focus the attack on the logical signal described by the checklist question."
DEFAULT_ASPECT_QUESTION_FOCUS = "Compares which side better satisfies the checklist question within this logical dimension."

ASPECT_ATTACK_RECIPES: Dict[str, str] = {
    "Task alignment & claim clarity (Task alignment & claim clarity)": (
        "- Goal: Weaken the answer's direct response to the core task so the central claim becomes less prominent, while keeping the topic and stance unchanged.\n"
        "- Recommended operations (choose 1-2; avoid collateral damage to other dimensions):\n"
        "  1) Move the explicit thesis from the opening or closing to a middle paragraph and soften the wording.\n"
        "  2) Add a short thematic preamble (1-2 sentences) at the beginning that does not answer the task directly, diluting task-alignment cues.\n"
        "  3) Split the single definitive conclusion into two tentative statements.\n"
        "- Prohibited operations:\n"
        "  * Do not change the overall stance or main conclusion.\n"
        "  * Do not introduce new topic branches or reorder paragraphs enough to disrupt global or local coherence.\n"
        "  * Do Not Add Key Facts.\n"
        "- Strength definition:\n"
        "  * low: make one change that reduces thesis prominence (for example, move or soften the claim).\n"
        "  * medium: weaken the thesis and add one light diluting preamble (no more than 2-3 new sentences in total).\n"
        "  * high: without changing the stance, add a mild distraction so readers must scan multiple paragraphs to reconstruct the core claim.\n"
    ),
    "Global coherence (Global coherence)": (
        "- Goal: Blur the macro-level progression so the main storyline feels looser, while preserving the core set of points.\n"
        "- Recommended operations (prioritize structural weakening over topic shifts):\n"
        "  1) Make one light paragraph reorder that slightly disrupts the task-driven sequence.\n"
        "  2) Remove or soften one paragraph-leading function sentence (for example, \"Below we analyze three aspects...\").\n"
        "  3) Downgrade the conclusion into a weaker recap so it echoes the body less strongly.\n"
        "- Prohibited operations:\n"
        "  * Do not introduce new threads or perspectives that drift away from the original stance.\n"
        "  * Do not delete large sections or rewrite the piece into a different genre.\n"
        "  * Do not add new facts.\n"
        "- Strength definition:\n"
        "  * low: weaken only one structural cue (no reordering).\n"
        "  * medium: perform one light reorder or weaken two structure-signaling sentences.\n"
        "  * high: combine one light reorder with a weaker conclusion echo while keeping the paragraph count essentially the same.\n"
    ),
    "Local coherence (Local coherence)": (
        "- Goal: Thin the logical bridges between adjacent sentences or paragraphs without changing the topic.\n"
        "- Recommended operations:\n"
        "  1) Delete or soften one to two key connective phrases or transition sentences (for example, \"therefore\", \"meanwhile\", \"by contrast\").\n"
        "  2) Rewrite an explanatory transition as a neutral side-by-side statement.\n"
        "  3) Insert one weak linking sentence that makes the flow feel forced while keeping the same topic.\n"
        "- Prohibited operations:\n"
        "  * Do not add new topic sections or perform large cross-paragraph reorders.\n"
        "  * Do not duplicate whole paragraphs or introduce obvious grammatical errors.\n"
        "  * Do Not Add Key Facts.\n"
        "- Strength definition:\n"
        "  * low: remove or weaken a single key transition.\n"
        "  * medium: weaken two transitions, or weaken one transition plus rewrite another into a flat statement.\n"
        "  * high: create one mild gap between paragraphs while still keeping the topic consistent.\n"
    ),
    "Evidence sufficiency & relevance (Evidence sufficiency & relevance)": (
        "- Goal: Leave the main argument framework intact but reduce how concrete, on-topic, or traceable the supporting evidence feels.\n"
        "- Recommended operations:\n"
        "  1) Replace one or two specific pieces of evidence (data, cases, mechanisms) with generalized statements such as \"studies suggest\" or \"some examples show\".\n"
        "  2) Remove one key supporting detail so the evidence feels thinner while still acknowledging its existence.\n"
        "  3) Keep the number of evidence points but make one slightly less aligned with the claim.\n"
        "- Prohibited operations:\n"
        "  * Do not fabricate new facts or add concrete numbers, years, or sources.\n"
        "  * Do not delete all evidence and leave the argument unsupported.\n"
        "- Strength definition:\n"
        "  * low: generalize a single evidence reference.\n"
        "  * medium: generalize one reference and delete one supporting detail.\n"
        "  * high: generalize two references or introduce one mildly off-target piece while avoiding new external facts.\n"
    ),
    "Warrants & causal reasoning (Warrants & causal reasoning)": (
        "- Goal: Keep the evidence and conclusions but erode the clarity of the \"why/how\" explanation without reversing the causal direction.\n"
        "- Recommended operations:\n"
        "  1) Retain the evidence and conclusion sentences but delete or soften one crucial mechanism explanation.\n"
        "  2) Recast a mechanism sentence as a vague inference (for example, \"this might imply\", \"it could therefore suggest\").\n"
        "  3) Compress a multi-step causal chain into a jumpy two-sentence \"because... therefore\" structure.\n"
        "- Prohibited operations:\n"
        "  * Do not reverse the causal direction or introduce new causal steps.\n"
        "  * Do not change the overall stance or conclusion.\n"
        "  * Do Not Add Key Facts.\n"
        "- Strength definition:\n"
        "  * low: weaken a single reasoning bridge.\n"
        "  * medium: weaken two reasoning bridges or compress a chain into a jumpy shortcut.\n"
        "  * high: remove one key intermediate explanation so the reader must infer the missing step.\n"
    ),
    "Qualifiers & counterpoints (Qualifiers & counterpoints)": (
        "- Goal: Remove the sense of caution, balance, or counterexample awareness so the argument reads more absolute while keeping the same stance.\n"
        "- Recommended operations:\n"
        "  1) Delete one or two qualifying phrases (for example, \"usually\", \"under certain conditions\", \"may\").\n"
        "  2) Delete one concession or risk-alert sentence (for example, \"although...\").\n"
        "  3) Replace cautious wording with firmer assertions without changing the core conclusion.\n"
        "- Prohibited operations:\n"
        "  * Do not add rebuttals that contradict the original stance or invent brand-new counterexamples.\n"
        "  * Do not invent new facts.\n"
        "- Strength definition:\n"
        "  * low: remove a single qualifier or hedging phrase.\n"
        "  * medium: remove one qualifier plus one concession or risk statement.\n"
        "  * high: rewrite the key conclusion paragraph in clearly absolute language while keeping the stance unchanged.\n"
    ),
    "Internal consistency (Internal consistency)": (
        "- Goal: Introduce mild but noticeable internal tension, such as shifting definitions, ranges, or conditions, without undermining the main claim entirely.\n"
        "- Recommended operations (favor definition or scope drift over numeric conflicts):\n"
        "  1) Use slightly different definitions or scopes for the same key concept at different points.\n"
        "  2) Offer subtly conflicting judgments about the same issue in separate paragraphs (for example, \"usually effective\" vs. \"often less effective\").\n"
        "  3) Create one inconsistency in stated conditions while avoiding new external facts.\n"
        "- Prohibited operations:\n"
        "  * Do not fabricate large volumes of new facts or obvious data errors.\n"
        "  * Do not make the contradiction so strong that it overturns the central claim outright.\n"
        "- Strength definition:\n"
        "  * low: introduce one mild drift in definition or condition.\n"
        "  * medium: introduce two mild inconsistencies (for example, definition + judgment, or judgment + condition).\n"
        "  * high: create one clearly identifiable contradictory pair without collapsing the overall stance.\n"
    ),
    "Concept introduction & logical transition (Concept introduction & logical transition)": (
        "- Goal: Disrupt the \"known -> new\" information flow so the opening or in-text concept introductions feel slightly abrupt.\n"
        "- Recommended operations:\n"
        "  1) Move one key background, definition, or setup sentence to a later position.\n"
        "  2) Introduce a term earlier but provide only a thin explanation (keep the topic recognizable).\n"
        "  3) Weaken paragraph-opening transitions so the information progression feels less natural.\n"
        "- Prohibited operations:\n"
        "  * Do not remove all background; avoid simultaneously tanking task alignment.\n"
        "  * Do not heavily alter evidence or causal content in other dimensions.\n"
        "- Strength definition:\n"
        "  * low: adjust the placement or clarity of one background/concept sentence.\n"
        "  * medium: introduce one concept early and under-explain another location.\n"
        "  * high: keep the same stance but create a two-point disturbance (a slightly abstract opening plus a delayed explanation later).\n"
    ),
}

DEFAULT_ATTACK_RECIPE = (
    "- Goal: Create a single, explainable weakness for {aspect}.\n"
    "- Recommended operations: keep edits minimal while targeting the logical signal described for this dimension.\n"
    "- Prohibited operations: do not damage other logical dimensions, add new facts, or change the stance."
)


def _parse_attack_strengths(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(DEFAULT_ATTACK_STRENGTHS)
    parts = [chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()]
    if not parts:
        parts = list(DEFAULT_ATTACK_STRENGTHS)
    invalid = [value for value in parts if value not in ATTACK_STRENGTH_CHOICES]
    if invalid:
        raise ValueError(
            f"Invalid attack strength(s): {invalid}. Allowed: {ATTACK_STRENGTH_CHOICES}"
        )
    return parts


class MissingAPIKeyError(RuntimeError):
    """Raised when the required API key is missing."""


class FilteredResponseError(RuntimeError):
    """Raised when the upstream model filters the response."""

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
                if now < self._next_time:
                    sleep_dur = self._next_time - now
                else:
                    self._next_time = now + self.interval
                    return
            time.sleep(min(sleep_dur, 5))


_RATE_LIMITERS: Dict[float, _RateLimiter] = {}
_RATE_LIMITERS_LOCK = threading.Lock()
_THREAD_LOCAL_CLIENT = threading.local()


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


def _configure_csv_field_size_limit() -> None:
    env_limit = os.environ.get("CSV_FIELD_SIZE_LIMIT")
    default_limit = 64 * 1024 * 1024  # 64 MB
    limit = default_limit
    if env_limit:
        try:
            limit = int(env_limit)
        except ValueError:
            pass
    try:
        csv.field_size_limit(limit)
    except OverflowError:
        csv.field_size_limit(2 ** 31 - 1)


_configure_csv_field_size_limit()


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_latin(char: str) -> bool:
    return ("A" <= char <= "Z") or ("a" <= char <= "z")


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "unknown"
    cjk = 0
    latin = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        elif _is_latin(ch):
            latin += 1
    total = cjk + latin
    if total == 0:
        return "other"
    if cjk >= max(1, int(total * 0.3)):
        return "zh"
    if latin >= max(1, int(total * 0.7)):
        return "en"
    if cjk > latin:
        return "zh"
    if latin > cjk:
        return "en"
    return "other"


def language_matches(text_a: str, text_b: str) -> bool:
    lang_a = detect_language(text_a)
    lang_b = detect_language(text_b)
    if lang_a == "unknown" or lang_b == "unknown":
        return True
    if lang_a == lang_b:
        return True
    if lang_a == "other" and lang_b == "other":
        return True
    return False


@dataclass
class ChecklistItem:
    index: int
    raw: Dict[str, Any]
    aspect: str
    question: str
    good_example: str
    bad_example: str
    span_hint: str


@dataclass
class SampleRecord:
    row: Dict[str, Any]
    sample_id: str
    label: str
    query: str
    answer_a: str
    answer_b: str
    checklist: List[ChecklistItem]


@dataclass
class AdversarialTask:
    sample: SampleRecord
    item: ChecklistItem
    favored_label: str
    favored_answer: str
    other_answer: str
    desired_attack_strength: str


@dataclass
class AdversarialResult:
    task: AdversarialTask
    response_text: str
    response_json: Dict[str, Any]
    created_at: str
    prompt_input: str


def _get_thread_client(config: Dict[str, Any]) -> AdversarialClient:
    client = getattr(_THREAD_LOCAL_CLIENT, "client", None)
    if client is None:
        client = AdversarialClient(**config)
        setattr(_THREAD_LOCAL_CLIENT, "client", client)
    return client


ADVERSARIAL_SYSTEM_PROMPT = (
    "You are a careful adversarial data designer. Modify the preferred answer only "
    "where instructed and produce JSON that exactly matches the requested schema."
)

ADVERSARIAL_USER_TEMPLATE = """
You are an adversarial sample construction expert who must transform the logically superior answer in this evaluation into a fragile variant that targets a specific checklist dimension.

[Task Background]
- QUERY: {query}
- Current better answer in evaluation ({favored_label}):\n{favored_answer}

[Target Logical Dimension]
- aspect: {aspect}
- definition: {aspect_definition}
- question focus: {aspect_question_focus}
- question: {question}
- good_example: {good_example}
- bad_example: {bad_example}
- span_hint: {span_hint}
- desired attack strength: {desired_attack_strength}

[Attack Recipe (custom guidance for this dimension)]
{attack_recipe}

[Construction Principles (all must be satisfied)]
1. Minimal Modification (Minimal Pair): Keep the adversarial answer closely aligned with the original in length, tone, and structure (length change ≤ 10%), and prioritize weakening by trimming or softening local reasoning.
2. Single-Dimension Targeted Damage (Dimension Isolation): Only degrade the target dimension above and preserve the original quality of the other seven logical dimensions.
3. Logical Damage First: Maintain fluent language, avoid grammatical errors or meaningless filler, and weaken by introducing argumentative flaws (vague evidence, missing qualifiers, broken causality, etc.).
4. Do Not Add Key Facts (Fact Stability): Do not invent new specific facts, data, or authorities. You may rewrite existing evidence into vaguer or less relevant statements.
5. Preserve Core Claims and Topics (Claim & Topic Preservation): Keep the theme and stance intact while weakening supporting chains, qualifiers, and information flow.
6. Strength Control: Preserve original paragraphs and formatting, and strictly limit the attack strength to {desired_attack_strength} according to the defined level of damage.
7. Maintain Style Consistency (Style Control): Do not rely on flashy formatting or heavy polishing to create artificial differences; only the logical chain should reveal the attack.
8. Human Verifiability (Human Verifiability): Ensure the adversarial flaws are immediately noticeable to human reviewers rather than being mistaken for random noise.

[Output Steps]
- First plan how to create obvious flaws in the target dimension while honoring the principles above.
- Describe the attack strategy and how it aligns with the target dimension.
- Explain how you keep other logical dimensions largely intact.

[Output Requirements]
Return a JSON object (do not include extra fields):
{
    "adversarial_answer": "Modified complete answer",
    "violation_rationale": "Explain in 2-3 sentences how you weakened this dimension",
    "preservation_notes": "Explain how other logical dimensions were preserved",
    "target_aspect": "{aspect}",
    "attack_strength": "{desired_attack_strength} (must match the instructions)"
}
""".strip()


class AdversarialClient:
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
        if not ANT_OPENAI_API_KEY_V2:
            raise MissingAPIKeyError("ANT_OPENAI_API_KEY_V2 is not set")
        try:  # Local import to avoid hard dependency at module import time
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency checked when needed
            raise ImportError(
                "The 'openai' package is required to generate adversarial samples. Please install it before running the script."
            ) from exc
        self.client = OpenAI(api_key=ANT_OPENAI_API_KEY_V2, base_url=BASE_URL)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout
        self.rate_limiter = _get_rate_limiter(requests_per_second)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
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

    def generate(self, task: AdversarialTask) -> AdversarialResult:
        attack_recipe = ASPECT_ATTACK_RECIPES.get(
            task.item.aspect,
            DEFAULT_ATTACK_RECIPE.format(aspect=task.item.aspect or "this dimension"),
        )
        aspect_meta = ASPECT_DEFINITIONS.get(task.item.aspect, {})
        aspect_definition = aspect_meta.get("definition", DEFAULT_ASPECT_DEFINITION)
        aspect_question_focus = aspect_meta.get("question_focus", DEFAULT_ASPECT_QUESTION_FOCUS)
        user_prompt = ADVERSARIAL_USER_TEMPLATE.format(
            query=task.sample.query,
            favored_label=task.favored_label,
            favored_answer=task.favored_answer,
            aspect=task.item.aspect,
            aspect_definition=aspect_definition,
            aspect_question_focus=aspect_question_focus,
            question=task.item.question,
            good_example=task.item.good_example,
            bad_example=task.item.bad_example,
            span_hint=task.item.span_hint,
            desired_attack_strength=task.desired_attack_strength,
            attack_recipe=attack_recipe,
        )

        messages = [
            {"role": "system", "content": ADVERSARIAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                call_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_completion_tokens": self.max_tokens,
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
                if not isinstance(parsed, dict):
                    raise ValueError("Model response did not contain a valid JSON object")
                return AdversarialResult(
                    task=task,
                    response_text=content,
                    response_json=parsed,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    prompt_input=user_prompt,
                )
            except Exception as exc:  # pragma: no cover - defensive robustness
                if self._is_filtered_error(exc):
                    raise FilteredResponseError("Adversarial generation filtered", code="1209") from exc
                last_error = exc
                print(f"AdversarialClient call failed on attempt {attempt}: {exc}")
                time.sleep(min(3 * attempt, 10))
        raise RuntimeError(f"Adversarial request failed after {self.max_attempts} attempts: {last_error}")


def _canonicalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return WINNER_NORMALIZATION_MAP.get(text, text.upper() if text in {"a", "b"} else text)


def _select_first_available(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _load_csv_rows(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    last_error: Optional[Exception] = None
    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            with open(path, "r", encoding=encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                fieldnames = list(reader.fieldnames or [])
                rows = [dict(row) for row in reader]
            break
        except UnicodeDecodeError as exc:
            last_error = exc
            rows = []
            continue
    else:
        raise RuntimeError(
            f"Unable to decode CSV {path} with encodings {CSV_ENCODING_CANDIDATES}"
        ) from last_error
    if rows and "sample_id" not in rows[0]:
        for idx, row in enumerate(rows):
            row.setdefault("sample_id", str(idx))
        if "sample_id" not in fieldnames:
            fieldnames = list(fieldnames) + ["sample_id"]
    return rows, fieldnames


def _parse_checklist(row: Mapping[str, Any]) -> List[ChecklistItem]:
    items: List[ChecklistItem] = []
    for idx in range(1, CHECKLIST_COUNT + 1):
        key = f"{CHECKLIST_PREFIX}{idx}"
        raw_value = row.get(key)
        if raw_value in (None, ""):
            return []
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            payload = raw_value
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {"aspect": "", "question": payload}
        if not isinstance(payload, dict):
            payload = {"aspect": "", "question": str(payload)}
        items.append(
            ChecklistItem(
                index=idx,
                raw=dict(payload),
                aspect=str(payload.get("aspect", OTHER_ASPECT_NAMES[idx - 1] if idx <= len(OTHER_ASPECT_NAMES) else "")),
                question=str(payload.get("question", "")),
                good_example=str(payload.get("good_example", "")),
                bad_example=str(payload.get("bad_example", "")),
                span_hint=str(payload.get("span_hint", "")),
            )
        )
    return items


def _identify_answer_columns(fieldnames: Sequence[str]) -> Tuple[str, str]:
    field_set = {name for name in fieldnames if isinstance(name, str)}
    for left, right in VALUE_PAIR_CANDIDATES:
        if left in field_set and right in field_set:
            return left, right
    raise RuntimeError("Could not locate answer columns (A/B) in CSV")


def load_samples(paths: Sequence[str], label_column: str) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for path in paths:
        rows, fieldnames = _load_csv_rows(path)
        answer_a_key, answer_b_key = _identify_answer_columns(fieldnames)
        for row in rows:
            enriched_row = dict(row)
            enriched_row.setdefault("source_csv", str(path))
            label = _canonicalize_label(enriched_row.get(label_column))
            if label not in {"A", "B"}:
                continue
            checklist = _parse_checklist(enriched_row)
            if len(checklist) != CHECKLIST_COUNT:
                continue
            query = _select_first_available(enriched_row, ["query", "question", "prompt", "QUERY"])
            answer_a = _select_first_available(enriched_row, [answer_a_key, "answer_a", "text_a", "response_a"])
            answer_b = _select_first_available(enriched_row, [answer_b_key, "answer_b", "text_b", "response_b"])
            if not query or not answer_a or not answer_b:
                continue
            sample_id = str(
                enriched_row.get("sample_id")
                or enriched_row.get("question_id")
                or enriched_row.get("id")
                or enriched_row.get("mid")
                or len(records)
            )
            records.append(
                SampleRecord(
                    row=enriched_row,
                    sample_id=sample_id,
                    label=label,
                    query=query,
                    answer_a=answer_a,
                    answer_b=answer_b,
                    checklist=checklist,
                )
            )
    if not records:
        raise SystemExit("No eligible samples found in CSVs (need label=A/B and 8 checklist items)")
    return records


def build_tasks(
    samples: List[SampleRecord],
    count: int,
    seed: Optional[int],
    strengths: Sequence[str],
) -> List[AdversarialTask]:
    rng = random.Random(seed)
    if count > len(samples):
        raise ValueError(f"Requested {count} samples but only {len(samples)} available")
    selected = rng.sample(samples, count)
    tasks: List[AdversarialTask] = []
    for sample in selected:
        favored_answer = sample.answer_a if sample.label == "A" else sample.answer_b
        other_answer = sample.answer_b if sample.label == "A" else sample.answer_a
        for item in sample.checklist:
            for strength in strengths:
                tasks.append(
                    AdversarialTask(
                        sample=sample,
                        item=item,
                        favored_label=sample.label,
                        favored_answer=favored_answer,
                        other_answer=other_answer,
                        desired_attack_strength=strength,
                    )
                )
    return tasks


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], *, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_key_component(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Collapse repeated whitespace to make keys stable across formatting differences.
    return " ".join(text.split())


def _normalize_strength(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    for strength in ATTACK_STRENGTH_CHOICES:
        if text.startswith(strength):
            return strength
    return text


def _build_unique_key(query: str, answer_a: str, answer_b: str, aspect: str, strength: str) -> str:
    parts = (
        _normalize_key_component(query),
        _normalize_key_component(answer_a),
        _normalize_key_component(answer_b),
        _normalize_key_component(aspect),
        _normalize_key_component(strength).lower(),
    )
    raw = "\u241F".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _task_resume_key(task: AdversarialTask) -> str:
    return _build_unique_key(
        task.sample.query,
        task.sample.answer_a,
        task.sample.answer_b,
        task.item.aspect,
        task.desired_attack_strength,
    )


def _extract_answer_pair_from_record(record: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    explicit_a = record.get("answer_a") or record.get("text_a")
    explicit_b = record.get("answer_b") or record.get("text_b")
    if explicit_a and explicit_b:
        return str(explicit_a), str(explicit_b)
    for left, right in VALUE_PAIR_CANDIDATES:
        if left in record and right in record:
            value_a = record.get(left)
            value_b = record.get(right)
            if value_a not in (None, "") and value_b not in (None, ""):
                return str(value_a), str(value_b)
    return None


def _extract_resume_key_from_record(record: Mapping[str, Any]) -> Optional[str]:
    existing_key = record.get("resume_key") or record.get("unique_key")
    if existing_key:
        return str(existing_key)
    query = record.get("query") or record.get("prompt") or record.get("QUERY")
    aspect = record.get("checklist_aspect") or record.get("aspect")
    strength = _normalize_strength(record.get("attack_strength") or record.get("desired_attack_strength"))
    answer_pair = _extract_answer_pair_from_record(record)
    if not (query and answer_pair and aspect and strength):
        return None
    answer_a, answer_b = answer_pair
    return _build_unique_key(str(query), answer_a, answer_b, str(aspect), str(strength))


def _load_existing_resume_keys(path: Path) -> Tuple[Dict[str, int], int]:
    key_counts: Dict[str, int] = {}
    total_records = 0
    if not path.exists():
        return key_counts, total_records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if existing_key_counts: 
                resume_skipped = 0 
                for idx in range(len(tasks) - 1, -1, -1): 
                    resume_key = _task_resume_key(tasks[idx]) 
                    if existing_key_counts.get(resume_key, 0) > 0: 
                        existing_key_counts[resume_key] -= 1 
                        resume_skipped += 1 
                        tasks.pop(idx) 
                if resume_skipped: 
                    print(f"Resume mode: skipped {resume_skipped} tasks; {len(tasks)} remaining to generate.") 

def _dedupe_jsonl_by_resume_key(path: Path) -> Optional[Dict[str, int]]:
    if not path.exists():
        return None
    records: List[Tuple[int, Optional[str], Optional[Dict[str, Any]], str]] = []
    last_index_by_key: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue
            record: Optional[Dict[str, Any]] = None
            resume_key: Optional[str] = None
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                record = None
            if isinstance(record, dict):
                resume_key = _extract_resume_key_from_record(record)
                if resume_key:
                    last_index_by_key[resume_key] = idx
            records.append((idx, resume_key, record, raw_line))
    if not records:
        return {
            "original_count": 0,
            "deduped_count": 0,
            "duplicates_removed": 0,
            "unique_key_count": 0,
        }
    indexes_to_keep = set()
    for idx, resume_key, _record, _raw_line in records:
        if not resume_key:
            indexes_to_keep.add(idx)
            continue
        if last_index_by_key.get(resume_key) == idx:
            indexes_to_keep.add(idx)
    if len(indexes_to_keep) == len(records):
        return {
            "original_count": len(records),
            "deduped_count": len(records),
            "duplicates_removed": 0,
            "unique_key_count": len(last_index_by_key),
        }
    temp_path = path.parent / f"{path.name}.dedupe_tmp"
    with temp_path.open("w", encoding="utf-8") as fh:
        for idx, _resume_key, _record, raw_line in records:
            if idx in indexes_to_keep:
                fh.write(raw_line + "\n")
    temp_path.replace(path)
    return {
        "original_count": len(records),
        "deduped_count": len(indexes_to_keep),
        "duplicates_removed": len(records) - len(indexes_to_keep),
        "unique_key_count": len(last_index_by_key),
    }


def _prune_jsonl_by_resume_keys(path: Path, allowed_keys: Set[str]) -> Optional[Dict[str, int]]:
    if not path.exists() or not allowed_keys:
        return None
    temp_path = path.parent / f"{path.name}.prune_tmp"
    total_records = 0
    removed_records = 0
    kept_records = 0
    records_without_key = 0
    matched_keys: Set[str] = set()
    with path.open("r", encoding="utf-8") as src, temp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue
            total_records += 1
            keep_line = True
            resume_key: Optional[str] = None
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                record = None
            if isinstance(record, dict):
                resume_key = _extract_resume_key_from_record(record)
            if resume_key:
                if resume_key not in allowed_keys:
                    keep_line = False
                    removed_records += 1
                else:
                    matched_keys.add(resume_key)
            else:
                records_without_key += 1
            if keep_line:
                dst.write(raw_line + "\n")
                kept_records += 1
    if removed_records == 0:
        temp_path.unlink(missing_ok=True)
        return {
            "original_count": total_records,
            "kept_count": kept_records,
            "removed_count": removed_records,
            "records_without_key": records_without_key,
            "allowed_key_count": len(allowed_keys),
            "matched_key_count": len(matched_keys),
            "missing_key_count": len(allowed_keys - matched_keys),
        }
    temp_path.replace(path)
    return {
        "original_count": total_records,
        "kept_count": kept_records,
        "removed_count": removed_records,
        "records_without_key": records_without_key,
        "allowed_key_count": len(allowed_keys),
        "matched_key_count": len(matched_keys),
        "missing_key_count": len(allowed_keys - matched_keys),
    }


def _generate_payload_for_task(
    task: AdversarialTask,
    client_config: Dict[str, Any],
    language_retry_limit: int,
) -> Optional[Dict[str, Any]]:
    client = _get_thread_client(client_config)
    model_name = client_config["model"]
    resume_key = _task_resume_key(task)
    for attempt in range(1, language_retry_limit + 1):
        try:
            result = client.generate(task)
        except FilteredResponseError as exc:
            print(
                f"Skip sample {task.sample.sample_id} checklist#{task.item.index} strength={task.desired_attack_strength}: model output was filtered ({exc.code})."
            )
            return None
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"Failed to generate adversarial sample (sample={task.sample.sample_id}, cklist={task.item.index}, strength={task.desired_attack_strength}): {exc}"
            )
            return None

        response_dict = result.response_json if isinstance(result.response_json, dict) else None
        if response_dict is None:
            print(
                f"Failed to generate adversarial sample (sample={task.sample.sample_id}, cklist={task.item.index}, strength={task.desired_attack_strength}): response missing JSON object"
            )
            continue

        adv_answer = response_dict.get("adversarial_answer")
        if not adv_answer:
            print(
                f"Failed to generate adversarial sample (sample={task.sample.sample_id}, cklist={task.item.index}, strength={task.desired_attack_strength}): response missing adversarial_answer"
            )
            continue
        if not language_matches(task.favored_answer, adv_answer):
            print(
                f"Language detection mismatch, retrying (sample={task.sample.sample_id}, cklist={task.item.index}, strength={task.desired_attack_strength}, attempt={attempt})"
            )
            if attempt == language_retry_limit:
                print(
                    f"Reached language retry limit; skip sample {task.sample.sample_id} checklist#{task.item.index} strength={task.desired_attack_strength}"
                )
            continue

        payload = dict(task.sample.row)
        payload.update(
            {
                "sample_id": task.sample.sample_id,
                "question_id": task.sample.row.get("question_id", task.sample.sample_id),
                "query": task.sample.query,
                "answer_a": task.sample.answer_a,
                "answer_b": task.sample.answer_b,
                "original_label": task.sample.label,
                "target_label": task.favored_label,
                "checklist_index": task.item.index,
                "checklist_aspect": task.item.aspect,
                "checklist_question": task.item.question,
                "adversarial_answer": adv_answer,
                "violation_rationale": response_dict.get("violation_rationale", ""),
                "preservation_notes": response_dict.get("preservation_notes", ""),
                "attack_strength": task.desired_attack_strength,
                "generator_model": model_name,
                "prompt_input": result.prompt_input,
                "response_text": result.response_text,
                "created_at": result.created_at,
                "resume_key": resume_key,
            }
        )
        return payload
    return None


def generate_adversarial_dataset(args: argparse.Namespace) -> None:
    try:
        attack_strengths = _parse_attack_strengths(args.attack_strengths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    language_retry_limit = max(1, args.language_retry_limit)
    chunk_size = max(1, args.write_chunk_size)
    samples = load_samples(args.input_csvs, args.label_column)
    tasks = build_tasks(samples, args.sample_count, args.seed, attack_strengths)
    task_resume_keys: Set[str] = {_task_resume_key(task) for task in tasks}

    client_config = {
        "model": args.generator_model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "requests_per_second": args.requests_per_second,
        "request_timeout": args.request_timeout,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume_mode = bool(getattr(args, "resume", False))
    dedupe_existing_output = bool(getattr(args, "dedupe_output", False))
    prune_mismatched_output = bool(getattr(args, "prune_mismatched_output", False))
    if dedupe_existing_output and not resume_mode:
        print("Warning: --dedupe-output is only meaningful with --resume; ignoring flag.")
        dedupe_existing_output = False
    if prune_mismatched_output and not resume_mode:
        print("Warning: --prune-mismatched-output is only meaningful with --resume; ignoring flag.")
        prune_mismatched_output = False
    if dedupe_existing_output and output_path.exists():
        dedupe_stats = _dedupe_jsonl_by_resume_key(output_path)
        if dedupe_stats:
            print(
                "Deduplication: compressed from {original} to {deduped} records, removed {removed} duplicates ({unique} unique resume_key values).".format(
                    original=dedupe_stats["original_count"],
                    deduped=dedupe_stats["deduped_count"],
                    removed=dedupe_stats["duplicates_removed"],
                    unique=dedupe_stats["unique_key_count"],
                )
            )
    if prune_mismatched_output and output_path.exists() and task_resume_keys:
        prune_stats = _prune_jsonl_by_resume_keys(output_path, task_resume_keys)
        if prune_stats:
            print(
                "Pruning: kept {kept}/{original} records (removed {removed}); task expects {allowed} target keys, matched {matched}, missing {missing}.".format(
                    kept=prune_stats["kept_count"],
                    original=prune_stats["original_count"],
                    removed=prune_stats["removed_count"],
                    allowed=prune_stats["allowed_key_count"],
                    matched=prune_stats["matched_key_count"],
                    missing=prune_stats["missing_key_count"],
                )
            )
    existing_key_counts: Dict[str, int] = {}
    if resume_mode:
        if output_path.exists():
            existing_key_counts, _ = _load_existing_resume_keys(output_path)
            if existing_key_counts:
                print(
                    "Resume mode: detected {records} existing records ({unique} unique keys); duplicate tasks will be skipped.".format(
                        records=sum(existing_key_counts.values()),
                        unique=len(existing_key_counts),
                    )
                )
        else:
            output_path.touch()
    else:
        output_path.write_text("", encoding="utf-8")

    resume_skipped = 0
    if resume_mode and existing_key_counts:
        filtered_tasks: List[AdversarialTask] = []
        for task in tasks:
            resume_key = _task_resume_key(task)
            if resume_key and existing_key_counts.get(resume_key, 0) > 0:
                existing_key_counts[resume_key] -= 1
                resume_skipped += 1
                continue
            filtered_tasks.append(task)
        tasks = filtered_tasks
        if resume_skipped:
            print(f"Resume mode: skipped {resume_skipped} tasks; {len(tasks)} remaining to generate.")

    if resume_mode and not tasks:
        print("Resume mode: all target tasks already exist; exiting immediately.")
        return

    pending_rows: List[Dict[str, Any]] = []
    total_written = 0
    num_workers = max(1, args.num_workers)
    total_tasks = len(tasks)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(
                _generate_payload_for_task,
                task,
                client_config,
                language_retry_limit,
            ): task
            for task in tasks
        }
        with tqdm(total=total_tasks, desc="Generating adversarial variants", unit="task") as bar:
            for future in as_completed(future_to_task):
                bar.update(1)
                task = future_to_task[future]
                try:
                    adversarial_payload = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    print(
                        f"Failed to generate adversarial sample (sample={task.sample.sample_id}, cklist={task.item.index}, strength={task.desired_attack_strength}): {exc}"
                    )
                    adversarial_payload = None
                if adversarial_payload:
                    pending_rows.append(adversarial_payload)
                    if len(pending_rows) >= chunk_size:
                        write_jsonl(output_path, pending_rows, mode="a")
                        total_written += len(pending_rows)
                        pending_rows.clear()

    if pending_rows:
        write_jsonl(output_path, pending_rows, mode="a")
        total_written += len(pending_rows)

    if total_written == 0:
        raise SystemExit("No adversarial samples were generated successfully")

    if resume_mode:
        print(f"Resume mode: appended {total_written} new adversarial samples to {args.output_path}")
    else:
        print(f"Wrote {total_written} adversarial variants to {args.output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct adversarial datasets from checklist-labelled CSVs")
    parser.add_argument(
        "--input-csv",
        dest="input_csvs",
        required=True,
        nargs="+",
        help="One or more CSV files to merge before sampling (same schema as vllm_infer inputs)",
    )
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Where to write the resulting JSONL")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT, help="Number of base samples to draw")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of concurrent worker threads for adversarial generation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN, help="Column name containing the preferred answer label")
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL, help="Model name to call via the judge API")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--requests-per-second", type=float, default=0.0, help="Optional rate limit for LLM calls")
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument(
        "--attack-strengths",
        default=",".join(DEFAULT_ATTACK_STRENGTHS),
        help="Comma-separated attack strengths to generate per sample (default: low,medium,high)",
    )
    parser.add_argument(
        "--language-retry-limit",
        type=int,
        default=DEFAULT_LANGUAGE_RETRY_LIMIT,
        help="Max retries when adversarial answer language mismatches the source",
    )
    parser.add_argument(
        "--write-chunk-size",
        type=int,
        default=10,
        help="Number of adversarial rows to buffer before appending to the JSONL output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output JSONL by skipping tasks sharing the same (query + answer_a + answer_b + aspect + attack strength) key",
    )
    parser.add_argument(
        "--dedupe-output",
        action="store_true",
        help="When using --resume, deduplicate by resume_key so only the most recent record per key remains",
    )
    parser.add_argument(
        "--prune-mismatched-output",
        action="store_true",
        help="When using --resume, trim the JSONL to keep only entries whose resume_key appears in this run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_adversarial_dataset(args)


if __name__ == "__main__":
    main()
