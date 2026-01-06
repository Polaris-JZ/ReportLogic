import argparse
import re

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import Levenshtein

# from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

def find_most_similar(target, candidates):
    # 初始化最小距离和最相似字符串
    min_distance = float('inf')
    most_similar = None
    
    # 遍历候选字符串列表
    for candidate in candidates:
        distance = Levenshtein.distance(target, candidate)  # 计算编辑距离
        if distance < min_distance:
            min_distance = distance
            most_similar = candidate
    
    return most_similar

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


class RewardModelProxy:
    def __init__(self, args):
        data_path = args.data_path
        # raw data dict of q2a
        with open(data_path, 'r') as f:
            all_data = [json.loads(line) for line in f.readlines()]
        self.raw_data_q2a = {}
        for data in all_data:
            question = data['question']
            answer = data['answer']
            # 可能是decode的问题，导致get_reward传进来的question可能缺少空格导致找不到KEY，所以这里把所有空格去除
            self.raw_data_q2a[question.replace(' ', '').replace('\n', '')] = answer
        
        self.tokenizer = get_tokenizer(
            args.reward_pretrain, None, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                r = self.get_math_reward(queries[i : min(len(queries), i + batch_size)])
                scores.extend(r)
        logger.info(f"scores: {scores}")
        return scores

    def math_verifier(self, sol, content):

        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        return verify(answer_parsed, gold_parsed)

    def get_math_reward(self, queries):
        scores = []
        for query in queries:
            # 需要去除chat template中的special token
            # 对于 llama
            if "<|end_header_id|>" in query or '<|eot_id|>' in query:
                question = query.split("<|start_header_id|>user<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0].strip()
                response = query.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0].strip()
            # 对于qwen
            elif '<|im_start|>' in query or '<|im_end|>' in query:
                question = query.split("<|im_start|>user")[-1].strip().split("<|im_end|>")[0].strip()
                response = query.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0]
            
            try:
                raw_answer = self.raw_data_q2a[question.replace(' ', '').replace('\n', '')]
            except KeyError:  # 兜底策略
                similar_question = find_most_similar(question.replace(' ', '').replace('\n', ''), list(self.raw_data_q2a.keys()))
                raw_answer = self.raw_data_q2a[similar_question]

            # 分为三种情况给分：精确匹配答案：1，指令遵循有boxed但是答案错误：0.1，指令不遵循：0
            if self.math_verifier(raw_answer, response):
                scores.append(1.0)
            elif 'boxed' in response:
                scores.append(0.1)
            else:
                scores.append(0.0)
        return scores


    def get_choice_reward(self, queries):
        scores = []
        for query in queries:
            # print(query)
            # 需要去除chat template中的special token
            # 对于 llama
            if "<|end_header_id|>" in query or '<|eot_id|>' in query:
                question = query.split("<|start_header_id|>user<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0].strip()
                response = query.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0].strip()
            # 对于qwen
            elif '<|im_start|>' in query or '<|im_end|>' in query:
                question = query.split("<|im_start|>user")[-1].strip().split("<|im_end|>")[0].strip()
                response = query.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0]
            # 可能是decode的问题，导致传进来的question可能缺少空格导致找不到KEY，所以这里把所有空格去除
            # try:
            raw_answer = self.raw_data_q2a[question.replace(' ', '').replace('\n', '')]
            # except:
            #     with open("/ossfs/workspace/OpenRLHF/error.txt", 'a') as f:
            #         f.write(query + '\n\n' + question + '=' * 100)
            #     scores.append(0.0)
            #     continue

            candidates = ["So the answer is", "The answer is", "答案选项为"]

            # 定义正则表达式模式
            pattern = r"(?:" + "|".join(re.escape(candidate) for candidate in candidates) + r")\s*([\w\s]+)[.,]?"
            # 查找raw answer所有匹配
            matches = re.findall(pattern, raw_answer, re.IGNORECASE)
            if matches:
                gold_answer = matches[-1].strip()
            else:
                scores.append(0.0)
                continue
            # 查找response的答案
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # 取最后一个匹配
                answer = matches[-1].strip()
                if answer == gold_answer:
                    scores.append(1.0)  # 指令遵循且答案正确
                else:
                    scores.append(0.1)  # 指令遵循但答案错误
            else:
                scores.append(0.0)  # 指令不遵循

        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    # Reference Dataset
    parser.add_argument("--data_path", type=str, default=None)

    args = parser.parse_args()

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
