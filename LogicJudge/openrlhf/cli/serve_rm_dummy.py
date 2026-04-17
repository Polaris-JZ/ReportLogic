import argparse
import re

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import re

# from openrlhf.models import get_llm_for_sequence_regression
# from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
        # raw data dict of q2a
        import os
        dataset_path = os.environ.get("RM_DATASET_PATH", "/path/to/data/rm_q2a_dataset.jsonl")
        with open(dataset_path, 'r') as f:
            all_data = [json.loads(line) for line in f.readlines()]
        self.raw_data_q2a = {}
        for data in all_data:
            question = data['messages'][0]['content']
            answer = data['messages'][1]['content']
            self.raw_data_q2a[question] = answer
        
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # # remove pad_token
        # for i in range(len(queries)):
        #     queries[i] = (
        #         strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
        #         + self.tokenizer.eos_token
        #     )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                # # dummy reward
                r = [1.0] * len(queries[i : min(len(queries), i + batch_size)])
                scores.extend(r)
        logger.info(f"scores: {scores}")
        return scores
    
    def get_choice_reward(self, queries):
        scores = []
        for query in queries:
            question = query.split("user")[-1].strip().split("assistant")[0].strip()
            response = query.split("assistant")[-1].strip()
            raw_answer = self.raw_data_q2a[question]

            candidates = ["So the answer is", "The answer is", "答案选项为"]

            # 定义正则表达式模式
            pattern = r"(?:" + "|".join(re.escape(candidate) for candidate in candidates) + r")\s*([\w\s]+)[.,]?"
            # 查找raw answer所有匹配
            matches = re.findall(pattern, raw_answer, re.IGNORECASE)
            gold_answer = matches[-1].strip()
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

    # def tokenize_fn(self, texts, device):
    #     batch = self.tokenizer(
    #         texts,
    #         return_tensors="pt",
    #         add_special_tokens=False,
    #         max_length=self.max_length,
    #         padding=True,
    #         truncation=True,
    #     )
    #     return {k: v.to(device) for k, v in batch.items()}


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
