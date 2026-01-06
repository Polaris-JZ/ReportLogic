import re
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import random
from tqdm import tqdm
import json

PROMPT = """**Role:** You are an efficient Mathematical Answer Verifier. Your task is to quickly evaluate a Model Answer (MA) against a Question (Q) and Gold Answer (GA) based on three key criteria: completeness, presence of irrelevant content, and final answer equivalence.

**Input:**

Question (Q):
%s

Gold Answer (GA):
%s

Model Answer (MA):
%s

**Evaluation Checks (Focus on these):**

1.  **Completeness:** Does the MA provide enough reasoning and a final answer to fully address the Question (Q)?
2.  **Irrelevant Content:** After presenting the core solution, does the MA contain substantial unrelated text, symbols, or chatter?
3.  **Answer Equivalence:** If the MA is complete, does its stated final answer match the Gold Answer (GA) mathematically?

**Output Format:**
You MUST output *only* the evaluation results as a single line using the following structured format, separated by '|':
`IsComplete: [YES/NO] | NoIrrelevant: [YES/NO] | IsEquivalent: [YES/NO] | Judgment: [YES/NO]`

**Judgment Rule:** Judgment is YES if and only if IsComplete=YES AND HasIrrelevant=NO AND IsEquivalent=YES.

**Verifier Output:**"""

IP_LIST = [
]

PORT_LIST = [
]

def reward_func(queries, prompts, labels):
    rewards = []
    formated_prompt_for_llm = []
    for query, prompt, label in zip(queries, prompts, labels):
        formated_prompt_for_llm.append(
            PROMPT % (
                prompt, 
                label,
                query.replace(prompt, "")
            )
        )
    llm_judge_list = process_list_with_threads(formated_prompt_for_llm)
    rewards = []
    for llm_judge in llm_judge_list:
        judge_res = extract(llm_judge)
        if judge_res and judge_res.get('Judgment', "") == 'YES':
            reward = 1.0
        else:
            reward = 0.0
        rewards.append(reward)
    return torch.tensor(rewards), llm_judge_list, formated_prompt_for_llm

def extract(text):
    pattern = r"IsComplete: (YES|NO) \| NoIrrelevant: (YES|NO) \| IsEquivalent: (YES|NO) \| Judgment: (YES|NO)"
    match = re.fullmatch(pattern, text)
    print(match)
    if match:
        is_complete = match.group(1)  # "Yes"
        has_irrelevant = match.group(2)  # "No"
        is_equivalent = match.group(3)  # "No"
        judgment = match.group(4)  # "NO"
        return {'IsComplete': is_complete, 'NoIrrelevant': has_irrelevant, 'IsEquivalent': is_equivalent, 'Judgment': judgment}
    else:
        return None


def process_list_with_threads(data_list, max_workers=256, timeout=600):
    # 初始化结果列表，长度与data_list相同
    results = [None] * len(data_list)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务并记录它们对应的索引
        future_to_index = {executor.submit(call_api, item): idx for idx, item in enumerate(data_list)}
        # 当每个任务完成时，将结果放入正确的位置
        for future in tqdm(as_completed(future_to_index), total=len(data_list), desc="Processing"):
        # for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e  # 或者你可以选择其他错误处理方式
    return results

def call_api(content):
    client = openai.Client(base_url=f"http://{random.choice(IP_LIST)}:{random.choice(PORT_LIST)}/v1", api_key="")
    response = client.chat.completions.create(
        model="VLLM_MODEL",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": content},
        ],
        stream=False,
        n=1,
        temperature=0.1,
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    import json
    data = json.loads(open("/path/to/llm_pretrain/exp/397432/OpenReasoner/orz_grpo_lite_Ling-Lite-base-OpenRLHF-Grouped_Gemm_lr1e-6_kl0_bs128_sample64_wo-think_sgtest_endboxed/rollout_output/Step1/Rank0.json").read())
    queries, prompts, labels = [], [], []
    for item in data:
        queries.append(item['prompt'] + item['response'])
        prompts.append(item['prompt'])
        labels.append(item['label'])
    
    rewards, llm_judge_list, formated_prompt_list = reward_func(queries, prompts, labels)
    cnt = 0 
    for reward, item, llm_judge, formated_prompt in zip(rewards, data, llm_judge_list, formated_prompt_list):
        if reward != item['reward']:
            print("formated prompt: ", formated_prompt)
            print("LLM Judge: ", llm_judge)
            print("Item: ", item)
            print("**"*10)
            cnt += 1
    print(f"{cnt=}")
    
