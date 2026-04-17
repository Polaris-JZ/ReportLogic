import re
import torch
import json
from itertools import islice, zip_longest
from transformers import AutoTokenizer
import re
from openrlhf.utils.utils import recover_original_order, distributed_order
import os

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
# GROUP_SIZE=8
# WORLD_SIZE=8

def extract_raw_prompt(prompt):
    # 定义正则表达式
    pattern = r'<role>HUMAN</role>(.*?)<role>ASSISTANT</role>'
    # 使用 re.search 查找匹配
    match = re.search(pattern, prompt, re.DOTALL)
    assert match is not None
    return match.group(1).strip()  # group(1) 是捕获组中的内容

# 准备vllm 推理内容
def vllm_reward_format_func(queries, prompts, labels, args):
    WORLD_SIZE = args.actor_num_nodes * args.actor_num_gpus_per_node
    GROUP_SIZE = args.n_samples_per_prompt
 
    queries = recover_original_order(WORLD_SIZE, queries)
    prompts = recover_original_order(WORLD_SIZE, prompts)
    labels = recover_original_order(WORLD_SIZE, labels)
    formated_prompt_for_llm = []
    
    label_json = json.loads(labels[0])
    if 'type' in label_json:
        GROUP_SIZE = 8
        rbs = int(len(queries) // GROUP_SIZE)
    else:
        rbs = int(len(queries) // GROUP_SIZE)
        assert args.rollout_batch_size == rbs


    for index in range(rbs):
        group_queries = queries[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        group_prompts = prompts[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        group_labels = labels[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        assert group_prompts[0]==group_prompts[-1]
        assert group_labels[0]==group_labels[-1]
        raw_prompt = extract_raw_prompt(group_prompts[0])
        label_json = json.loads(group_labels[0])

        for i in range(GROUP_SIZE):
            for j in range(i+1, GROUP_SIZE):
                rsp_i = group_queries[i].replace(group_prompts[i], "").strip().strip("<|endoftext|>")
                if 'type' in label_json and label_json['type'] == 'eval':
                    rsp_j = label_json['ground_truth_response']
                else:
                    rsp_j = group_queries[j].replace(group_prompts[j], "").strip().strip("<|endoftext|>")
                fomated_prompt = prompt_template.format(dialog=raw_prompt, response1=rsp_i, response2=rsp_j)
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
    print(f"{len(formated_prompt_for_llm)=}")
    return formated_prompt_for_llm
            
def extract_score(text):
    try:
        match = re.search(r'\[([\d.]+),\s*([\d.]+)\]\s*$', text)
        if match:
            score1_str = match.group(1)
            score2_str = match.group(2)
            
            rsp_i_score = float(score1_str) if '.' in score1_str else int(score1_str)
            rsp_j_score = float(score2_str) if '.' in score2_str else int(score2_str)
        else:
            rsp_i_score, rsp_j_score = 0, 0
    except Exception as e:
        rsp_i_score, rsp_j_score = 0, 0

    score_diff = rsp_i_score - rsp_j_score
    if score_diff >= 0.5-0.001:
        return 1.0, -1.0
    elif (score_diff<=0.5+0.001) and (score_diff>=-0.5-0.001):
        return 0.0, 0.0
    else:
        assert score_diff <= -0.5
        return -1.0, 1.0

def vllm_reward_post_process_func(vllm_reward_responses, queries, prompts, labels, args):
    """
    计算reward值
    :param vllm_reward_responses: reward_vllm_engines output
    :param queries: prompt + response
    :param prompts: prompt
    :param labels: 标签
    :return: reward值，需要为tensor float类型

    在这里组装prompts，然后输入到generate_func中，获取生成式模型的输出
    """
    WORLD_SIZE = args.actor_num_nodes * args.actor_num_gpus_per_node
    GROUP_SIZE = args.n_samples_per_prompt
 
    queries = recover_original_order(WORLD_SIZE, queries)
    prompts = recover_original_order(WORLD_SIZE, prompts)
    labels = recover_original_order(WORLD_SIZE, labels)
    label_json = json.loads(labels[0])
    if 'type' in label_json:
        GROUP_SIZE = 8
        rbs = int(len(queries) // GROUP_SIZE)
    else:
        rbs = int(len(queries) // GROUP_SIZE)
        assert args.rollout_batch_size == rbs

    rewards = []
    for index in range(rbs):
        group_queries = queries[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        group_prompts = prompts[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        group_labels = labels[index * GROUP_SIZE: (index+1) * GROUP_SIZE]
        GROUP_SIZE_VLLM = int(len(vllm_reward_responses) // rbs)
        group_vllm_reward_responses = vllm_reward_responses[index * GROUP_SIZE_VLLM: (index+1) * GROUP_SIZE_VLLM]


        assert group_prompts[0]==group_prompts[-1]
        assert group_labels[0]==group_labels[-1]
        raw_prompt = extract_raw_prompt(group_prompts[0])
        label_json = json.loads(group_labels[0])

        group_rewards = [[] for _ in range(GROUP_SIZE)]
        vllm_index = 0
        for i in range(GROUP_SIZE):
            for j in range(i+1, GROUP_SIZE):
                vllm_reward_response = group_vllm_reward_responses[vllm_index]
                vllm_index += 1
                rsp_i_score, rsp_j_score = extract_score(vllm_reward_response)
                if 'type' in label_json and label_json['type'] == 'eval':
                    # For evaluation
                    group_rewards[i].append(rsp_i_score)
                    group_rewards[j].append(0.0)
                else:
                    # For train
                    group_rewards[i].append(rsp_i_score)
                    group_rewards[j].append(rsp_j_score)
        assert vllm_index == len(group_vllm_reward_responses)
        for gr in group_rewards:
            rewards.append(sum(gr))
    rewards = distributed_order(WORLD_SIZE, rewards)
    return torch.tensor(rewards).to(torch.float32)

        
