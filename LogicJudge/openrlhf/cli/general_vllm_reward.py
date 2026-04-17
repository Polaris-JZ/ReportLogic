import os
import re
import torch
import json
from itertools import islice, zip_longest
from transformers import AutoTokenizer

path = os.environ.get("REWARD_TOKENIZER_PATH", "/path/to/checkpoint/reward-tokenizer")
tokenizer = AutoTokenizer.from_pretrained(path)

prompt_template = """你是一位熟练的评分小能手。你应该根据给定的用户问题生成全面丰富细致的评判标准，并对智能体回复内容进行评估。

#### 评估步骤 ####
首先，根据对话上下文，列举出需要对回复内容进行评估的多个维度、权重（总和100%），维度越丰富越全面越细致对最终的评估越有利。给出每个维度下列出符合程度的判断标准，例如，1 分表示答案完全不符合标准，3 分表示答案仅少部分符合标准，7分表示大部分答案符合标准，10 分表示答案完全符合评估标准。
然后，根据智能助理的回复和评估标准，针对每个标准给出1-10的评分及详细的理由，分数越高，答案越符合相关标准。
最后，对各个标准下的分数进行加权求和计算总分，总分在1-10之间。

#### 输出格式要求 ####
输出三行
具体标准：<针对查询和上下文的其他潜在标准，以及每个标准的权重和符合程度判断标准>
分析：<根据给定标准分析不同的回复内容>
  [第1个答案分析]
  [第2个答案分析]
得分：<按顺序显示所有回复的综合得分，在方框内用逗号分隔，[x,x]>

#### 对话上下文 ####
{dialog}

#### 待评分答案 ####
[第1个答案开头]
{response1}
[第1个答案结尾]

[第2个答案开头]
{response2}
[第2个答案结尾]

评分前，请逐步分析。你的评分需要尽可能严格客观，注意回复内容并非越长代表质量越好，过于冗长的回复不一定是更好的回复，请从回复质量上进行判断。
"""
def extract_raw_prompt(prompt):
    # 定义正则表达式
    pattern = r'<role>HUMAN</role>(.*?)<role>ASSISTANT</role>'
    # 使用 re.search 查找匹配
    match = re.search(pattern, prompt, re.DOTALL)
    assert match is not None
    return match.group(1).strip()  # group(1) 是捕获组中的内容

# 准备vllm 推理内容
def vllm_reward_format_func(queries, prompts, labels):

    formated_prompt_for_llm = []
    for query, prompt, label in zip(queries, prompts, labels):
        # 解析 label 字段
        label_json = json.loads(label)
        raw_prompt = extract_raw_prompt(prompt)
        model_response = query.replace(prompt, "").strip().strip("<|endoftext|>")
        groud_truth_response = label_json['ground_truth_response']
        fomated_prompt = prompt_template.format(dialog=raw_prompt, response1=model_response, response2=groud_truth_response)
        formated_prompt_for_llm.append(
            tokenizer.apply_chat_template(
                [{
                    'role': 'user',
                    'content': fomated_prompt
                }],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        )
    return formated_prompt_for_llm

def extract_score(text):
    try:
        match = re.search(r'\[([\d.]+),\s*([\d.]+)\]\s*$', text)
        if match:
            score1_str = match.group(1)
            score2_str = match.group(2)
            
            score1 = float(score1_str) if '.' in score1_str else int(score1_str)
            score2 = float(score2_str) if '.' in score2_str else int(score2_str)
            
            return score1, score2  
        else:
            return 0, 0
    except Exception as e:
        return 0, 0

def vllm_reward_post_process_func(vllm_reward_responses, queries, prompts, labels):
    """
    计算reward值
    :param vllm_reward_responses: reward_vllm_engines output
    :param queries: prompt + response
    :param prompts: prompt
    :param labels: 标签
    :return: reward值，需要为tensor float类型

    在这里组装prompts，然后输入到generate_func中，获取生成式模型的输出
    """

    rewards = []
    for vllm_reward_response in vllm_reward_responses:
        response_score, gt_score = extract_score(vllm_reward_response)
        if response_score==0 and gt_score==0:
            rewards.append(-100)
        else:
            rewards.append(response_score-gt_score)
    return torch.tensor(rewards).to(torch.float32)
 

if __name__ == "__main__":
    sample_path = os.environ.get("SAMPLE_DATA_PATH", "/path/to/data/sample.jsonl")
    tokenizer_path = os.environ.get("CHAT_TOKENIZER_PATH", "/path/to/checkpoint/chat-tokenizer")
    data = [json.loads(line) for line in open(sample_path).readlines()[: 100]]
    chat_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    queries, prompts, labels = [], [], []
    for example in data:
        label_json = json.loads(example['label'])
        prompt = chat_tokenizer.apply_chat_template(
            [
                {'role': 'user', 'content': example['input']}
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        queries.append(prompt+label_json['ground_truth_response'])
        labels.append(example['label'])
    print(vllm_reward_format_func(queries, prompts, labels)[0])
    print(vllm_reward_post_process_func(["[0, 0]","[1, 0]"], queries, prompts, labels))
    