import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from tqdm import tqdm

import bailing_verifier.verifiers
from bailing_verifier.registry import VERIFIER_FACTORY
from bailing_verifier.verifiers.code.sandbox.ant_sandbox import ant_sandbox_verifier

# Todo 代码Sandbox需要多线程并发，但是math_verify不能使用多线程，这里需要处理一下
def reward_func(queries, prompts, labels):
    # 对数据进行分类: 沙盒数据与非沙盒数据
    sandbox_data_index, sandbox_data, other_data , other_data_index = [], [], [], []

    for index, query, prompt, label in zip(range(len(queries)), queries, prompts, labels):
        if 'sandbox' in json.loads(label)['verifier']:
            sandbox_data_index.append(index)
            sandbox_data.append([query, prompt, label])
        else:
            other_data_index.append(index)
            other_data.append([query, prompt, label])
    
    sandbox_verify_results = process_list_with_threads(sandbox_data)
    other_verify_results = [call_verify(item) for item in other_data]

    # 合并结果
    overall_verify_results = [None] * len(queries)
    for index, verify_judge in zip(sandbox_data_index, sandbox_verify_results):
        overall_verify_results[index] = verify_judge
    for index, verify_judge in zip(other_data_index, other_verify_results):
        overall_verify_results[index] = verify_judge
    
    rewards = []
    for verifier_judge in overall_verify_results:
        if verifier_judge:
            reward = 1.0
        else:
            reward = 0.0
        rewards.append(reward)
    return torch.tensor(rewards)

def process_list_with_threads(data_list, max_workers=256, timeout=600):
    # 初始化结果列表，长度与data_list相同
    results = [None] * len(data_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务并记录它们对应的索引
        future_to_index = {executor.submit(call_verify, item): idx for idx, item in enumerate(data_list)}
        # 当每个任务完成时，将结果放入正确的位置
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e  # 或者你可以选择其他错误处理方式
    return results


def call_verify(item):
    query, prompt, label = item # query: 输入+输出 prompt: 输入 label: 自定义字段
    data_dict = json.loads(label)
    verification_func = VERIFIER_FACTORY.get(data_dict['verifier'])
    # verification_func = ant_sandbox_verifier
    model_prediction = query.replace(prompt, "")
    testcases = data_dict["gold"]
    verify_result = verification_func(model_prediction, testcases)
    flag = verify_result['pass']
    return flag



if __name__ == '__main__':
    # queries, prompts, labels = [], [], []
    # max_test = 0
    # for line in open("/path/to/llm_pretrain/dataset/lite_rl_data/code_rl_data/codeforces_eve_6k_gpt4o_results.jsonl").readlines()[:256]:
    #     line = json.loads(line)
    #     queries.append(line['input'] + "\n" + line['gpt4o-1120'])
    #     prompts.append(line['input'])
    #     labels.append(json.dumps(
    #         {
    #             'gold': line['gold'],
    #             'verifier': line['verifier'],
    #         }
    #     ))
    #     if len(json.loads(line['gold'])['inputs']) > max_test:
    #         max_test = len(json.loads(line['gold'])['inputs'])
    # print(max_test)

    queries, prompts, labels = [], [], []
    path = "/path/to/llm_pretrain/exp/397432/tmp/OpenRLHF_debug/load_debug/rollout_output/Step3/Rank0.json"
    for line in json.loads(open(path, "r").read()):
        queries.append(line['prompt'] + "\n" + line['response'])
        prompts.append(line['prompt'])
        labels.append(line['label'])
    rewards = reward_func(queries, prompts, labels)
    print(rewards)
