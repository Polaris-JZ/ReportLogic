import json
import re
from typing import Dict

import torch

try:
    from jsonschema import Draft7Validator, ValidationError
except ImportError as exc:
    raise ImportError(
        "jsonschema package is required for think-format validation. Install via 'pip install jsonschema'."
    ) from exc

CHOICES = ["A>B", "A<B", "both good", "both bad"]

REQUIRED_ASPECT_KEYWORDS = [
    "任务对齐与主张清晰",
    "全局连贯性",
    "局部连贯性",
    "证据充分性与相关性",
    "推理桥与因果链条",
    "限定与反驳处理",
    "内部一致性",
    "概念引入与逻辑衔接",
]


def _aspect_schema():
    return {
        "type": "object",
        "required": ["winner", "explanation"],
        "properties": {
            "winner": {"type": "string", "enum": CHOICES},
            "explanation": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }


THINK_SCHEMA = {
    "type": "object",
    "required": ["aspect_evaluations", "overall_explanation"],
    "properties": {
        "aspect_evaluations": {
            "type": "object",
            "minProperties": len(REQUIRED_ASPECT_KEYWORDS),
            "additionalProperties": _aspect_schema(),
        },
        "overall_explanation": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}

THINK_VALIDATOR = Draft7Validator(THINK_SCHEMA)


def _covers_required_keywords(aspects: Dict[str, Dict[str, str]]) -> bool:
    coverage = {kw: False for kw in REQUIRED_ASPECT_KEYWORDS}
    for key in aspects:
        for kw in REQUIRED_ASPECT_KEYWORDS:
            if kw in key:
                coverage[kw] = True
    return all(coverage.values())


def check_think_format(raw_think: str) -> bool:
    """Ensure think block is a JSON dict with required schema."""
    try:
        think_obj = json.loads(raw_think)
    except json.JSONDecodeError:
        return False

    try:
        THINK_VALIDATOR.validate(think_obj)
    except ValidationError:
        return False

    aspects = think_obj.get("aspect_evaluations", {})
    if not _covers_required_keywords(aspects):
        return False

    return True

def check_output_format(text):
    analysis = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    pred = re.findall(r'</think>\n\n(.*?)<\|im_end\|>', text, re.DOTALL)

    if len(analysis) != 1:
        return None

    think_block = analysis[0].strip()
    if not think_block or not check_think_format(think_block):
        return None

    if len(pred) != 1:
        return None

    candidate = pred[0].strip()
    if candidate in CHOICES:
        return candidate

    return None

def get_reward(query, prompt, label):
    """
    query:
        <|im_start|>user
        <prompt><|im_end|>
        <|im_start|>assistant
        <think><analysis></think>
        
        <result><|im_end|>
    prompt:
        <|im_start|>user
        <prompt><|im_end|>
        <|im_start|>assistant

    label:
        <result>
    """
    output = query.replace(prompt, '')
    valid_output = check_output_format(output)
    if valid_output is None:
        return -1.0
    
    if valid_output == label:
        return 1.0
    else:
        return -0.5

def reward_func(queries, prompts, labels):
    return torch.tensor([
        get_reward(query, prompt, label)
        for query, prompt, label in zip(queries, prompts, labels)
    ])

if __name__ == '__main__':
    queries = [
        '<|im_start|>user\n\n<prompt><|im_end|>\n<|im_start|>assistant\n\n<think><analysis></think>\n\nA<B<|im_end|>',
        '<|im_start|>user\n\n<prompt><|im_end|>\n<|im_start|>assistant\n\n<analysis></think>\n\nA=B<|im_end|>',
        '<|im_start|>user\n\n<prompt><|im_end|>\n<|im_start|>assistant\n\n<think><analysis></think>\n\nA<B<|im_end|>',
    ]
    prompts = [None] * len(queries)
    labels = ['A<B', 'A=B', 'A>B']
    print(reward_func(queries, prompts, labels))