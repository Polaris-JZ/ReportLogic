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

ANT_OPENAI_API_KEY_V2 = os.environ.get("ANT_OPENAI_API_KEY_V2")
BASE_URL = os.environ.get("ANT_OPENAI_BASE_URL", "https://matrixllm.alipay.com/v1/")
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
You are a logic analysis expert。

Your task is：Based on the question QUERY and two answer texts（A and B），generate a **logic evaluation checklist（checklist）**，to help annotators compare the two answers across eight logical dimensions。

——

## Three-Level Logical Framework Description
This evaluation is based on classical logic and argumentation analysis theory，dividing text logic into three levels：
1. **Macro level（Macro level）**：focuses on the overall logical direction and thematic focus of the text，determining whether the answer matches the task and maintains global consistency。
2. **Structural level（Structural level）**：examines the completeness of the argumentative chain，including claims, evidence, reasoning, qualifiers, and consistency。
3. **Expositional level（Expositional level）**：combining information flow theory（Given→New），evaluating whether the development of information from given to new is natural, smooth, and easy to understand。

——

## Eight Logical Evaluation Dimensions Descriptionand Question Focus on重点


1. **Task alignment & claim clarity（Task alignment & claim clarity）**  
   - **Definition**：Focuses on whether the answer（如开头或结尾）清楚responds to了 QUERY 's core task at a prominent position，是否提出了明确、集中的中心论点。Evaluates whether the entire text stays focused on the topic，and whether there are digressions or deviations。  
   - **Question Focus on**：Compares which side“more clearly responds to the question at the beginning or end、presents a central position、maintains thematic consistency”。

2. **Global coherence（Global coherence）**  
   - **Definition**：Focus on回答整体从头到尾的逻辑架构是否清晰、合理，内容是否按照符合 QUERY 要求的逻辑顺序展开（例如：若 QUERY 要求分析，应体现背景→问题→分析→结论；若 QUERY 要求比较，应体现对比→评价→结论等）。重点在于全篇是否形成稳定的逻辑主线。  
   - **Question Focus on**：Compares which side“has a more organized overall structure、follows a development sequence that better aligns with task logic、has conclusions that better echo the preceding text”。

3. **Local coherence（Local coherence）**  
   - **Definition**：Focus onwhether the logical relationships between adjacent paragraphs or sentences are coherent，whether there are reasonable transitions or connections between arguments。focuses on judging whether paragraph transitions are smooth、and whether there are logical breaks or jumps。  
   - **Question Focus on**：Compares which side“段落衔接更合理、逻辑过渡更连贯”。

4. **Evidence sufficiency & relevance（Evidence sufficiency & relevance）**  
   - **Definition**：Focus on关键主张是否由充分且贴题的证据支撑（如数据、实验、事实、文献、案例等），以及这些证据and结论之间是否形成清晰、可追溯的逻辑关系。  
   - **Question Focus on**：Compares which side“提供的证据更具体、更贴题、更能支撑主要结论”。

5. **Warrants & causal reasoning（Warrants & causal reasoning）**  
   - **Definition**：Focus on回答是否清楚说明“为什么”这些证据能支撑结论，推理过程是否完整、因果方向是否合理，是否存在逻辑跳跃。  
   - **Question Focus on**：Compares which side“推理链更完整、因果关系更清晰、逻辑过渡更合理”。

6. **Qualifiers & counterpoints（Qualifiers & counterpoints）**  
   - **Definition**：Focus on回答是否识别并处理潜在反例、不确定性或例外情况，是否通过限定条件、让步或平衡表述避免绝对化论断。  
   - **Question Focus on**：Compares which side“更能识别反例、设置前提或限定条件，使论证更平衡”。

7. **Internal consistency（Internal consistency）**  
   - **Definition**：Focus on回答内部的数据、Definition和立场是否前后一致，有无比例矛盾、Definition变化或自相矛盾的论点。  
   - **Question Focus on**：Compares which side“论述更前后一致、内部逻辑更自洽”。

8. **Concept introduction & logical transition（Concept introduction & logical transition）**  
   - **Definition**：该维度Focus on文本中信息展开的自然性，尤其是从“已知 → 新知”的过渡是否平滑。  
     评估顺序分两级：  
     1. **篇首层面**：文章开头是否先交代必要的背景、上下文或问题缘起，再自然过渡到主要议题或论点。  
        - 若一开头就直接跳入抽象讨论或使用读者不熟悉的术语，则视为逻辑衔接不足。  
     2. **篇内层面**：在正文中引入新概念、模型或术语时，是否提供了简短解释或上下文，使读者能理解其作用和来源。  
   - **Question Focus on**：  
     优先Compares which side“在文章开头更好地铺陈背景并自然引出主题”；若二者开头都合理，再Compares which side“在正文中引入新概念时过渡更自然、解释更充分”。

——

## 输入格式

[INPUT]
- QUERY
<<<QUERY>>>

- TEXT A
<<<TEXT_A>>>

- TEXT B
<<<TEXT_B>>>

——

## Output Requirements
- 仅输出 **JSON 数组（长度 = 8）**，数组中每个对象对应上述 1–8 的一个方面，且顺序一致。
- 每个对象必须包含以下 5 个字段：  
  - `"aspect"`：上述八项之一（采用精确命名）  
  - `"question"`：该维度下、针对当前 QUERY + 回答 A/B 的**具体判断问题**。  
    - 问题必须结合当前语境，指向实际可比的逻辑现象，不能抽象或笼统（如“是否合理”“是否充分”等）。  
    - 应生成让评审者可以直接在 A/B 中判断“哪一方更好”的具体问题。  
  - `"good_example"`：and该 question 直接对应的良好回答示例，说明“如果做得好，会是什么样”。  
  - `"bad_example"`：and该 question 直接对应的差表现示例，说明“如果做得差，会出现什么问题”。  
    - 两者都必须基于同一 question，反映该维度下同一逻辑点的好坏表现。  
  - `"span_hint"`：提示标注员应Focus on的语言信号或逻辑线索，需结合当前回答自动定制。  


——

## 生成指引
- Checklist 必须**根据每题内容定制生成**，不可使用空洞或模板化表达。  
- `question` 必须具备“语义自足 + 逻辑焦点 + 明确比较对象”三要素。  
  具体来说：
  1. **必须点明被比较的主张或主题对象**（如“碳税政策减少排放”“大型语言模型提升推荐效果”），不能只说“关键主张”或“模型假设”。  
  2. **必须说明要比较的逻辑焦点**（如“证据的具体性and贴题性”“因果链的完整性”“段落衔接的合理性”），不能只说“更合理”“更清晰”。  
  3. **必须用自然语言让评审者脱离上下文也能理解问题在问什么。**
     - 不要写：“哪一方对关键主张提供更具体支撑？”
     - 写：“比较 A/B，哪一方对‘碳税政策减少排放’这一主张提供了更具体的证据（如排放数据、案例、机制说明），并解释这些证据如何支撑结论？”
  4. 禁止使用仅列术语或专业符号的括号式说明（如“MVO=方差风险”），除非它直接出现在原文并且被Definition过。
  5. 如果原文的主张过多或难界定，请选择文中**最核心、最反复强调的主张**作为问题焦点。
- 对于第5项的问题生成（Warrants & causal reasoning），需额外遵循：
  - `question` 必须明确指出 A/B 所论述的具体逻辑链或推理方向，而非笼统的“是否合理”。  
  - 禁止使用符号式或箭头式的因果串联（如“X→Y→Z”）；应改写为自然语言句式（如“X 导致 Y，进而影响 Z”）。  
  - 问题应包含：
    1. **因果对象**（要解释的核心主张，如“政策创新促进房地产可持续发展”）；  
    2. **因果路径**（解释“为什么”或“如何”成立）；  
    3. **比较焦点**（哪一方更清晰或更合理）。  
  - 示例：
    - 不佳： “哪一方更清楚阐明‘为何这些指标能支撑准确比较’的因果链条？”（过于抽象）
    - 优化： “比较 A/B，哪一方更清楚地解释了‘为什么政策创新、市场需求升级、技术进步能够成为房地产可持续发展的动力’的因果机制？”
- `good_example` and `bad_example` 必须and question 严格对应，是同一判断点的正反体现。  
- 仅评估**逻辑结构and论证质量**，不涉及事实正确性或写作风格。  
- 所有输出使用自然、简洁、可读性强的中文。  
- 输出中不得出现评分或总结。  

——

## 输出示例（展示风格）
[
  {
    "aspect": "Evidence sufficiency & relevance（Evidence sufficiency & relevance）",
    "question": "比较 A/B，哪一方对‘碳税政策减少排放’这一主张提供了更具体的证据（如排放数据、国家案例、经济机制说明），并解释这些证据如何支撑结论？",
    "good_example": "引用具体国家数据或研究结果，并说明碳税如何促使企业减少排放，从而支持论点。",
    "bad_example": "仅笼统声称‘碳税有效’，缺少数据或逻辑说明。",
    "span_hint": "Focus on‘数据显示/研究发现/案例表明’等证据信号and结论衔接。"
  },
  ...
]

——

请严格遵循以上约束，仅输出符合键名and结构的 JSON 数组（长度=8），所有描述性内容必须使用中文。
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
        if not ANT_OPENAI_API_KEY_V2:
            raise MissingAPIKeyError("ANT_OPENAI_API_KEY_V2 is not set")
        self.client = OpenAI(api_key=ANT_OPENAI_API_KEY_V2, base_url=BASE_URL)
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
        print(f"跳过题目 {task.question_id}：评审结果被过滤 ({exc.code})")
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
        print(f"跳过题目 {task.question_id}：连续失败 {params.get('max_attempts', 0)} 次")
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
