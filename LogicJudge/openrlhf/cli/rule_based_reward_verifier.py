import torch
from openrlhf.cli.ifeval_eval import get_ifeval_reward
from openrlhf.cli.serve_rm_math_boxed import get_correct_reward as get_boxed_reward
import json
import bailing_verifier.verifiers
from bailing_verifier.registry import VERIFIER_FACTORY
from bailing_verifier.verifiers.code.sandbox.ant_sandbox import ant_sandbox_verifier
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import ray
from multiprocessing import Pool

if not ray.is_initialized():
    ray.init()


def aggregation_rm_func(rewards, query, prompt, label):
    return rewards[0] * rewards[1]


def get_verifier_reward(query, prompt, label):
    data_dict = json.loads(label)
    if 'autoif' in data_dict['verifier']:
        data_dict['verifier'] = 'assert-autoif-verifier'
    verification_func = VERIFIER_FACTORY.get(data_dict['verifier'])
    model_prediction = query.replace(prompt, "").replace('<|endoftext|>', '')
    testcases = data_dict["gold"]

    if "template_answer" in data_dict:
        verify_result = verification_func(model_prediction, testcases, template_answer=data_dict["template_answer"])
    else:
        verify_result = verification_func(model_prediction, testcases)
    flag = verify_result['pass']
    if flag:
        return 1.0
    else:
        return 0.0


@ray.remote(num_cpus=1)
def get_reward(item):
    query, prompt, label = item
    try:
        label_json = json.loads(label)
        label_is_dict = True
    except Exception as e:
        label_json = None
        label_is_dict = False

    try:
        if not label_is_dict:
            return get_boxed_reward(query, prompt, label)  # math verify reward

        # IFEval
        elif 'key' in label and 'instruction_id_list' in label and 'prompt' in label and 'kwargs' in label:
            return get_ifeval_reward(query, prompt, label)  # 确定为IFEval的evaluation结果
        
        # bailing_verifier
        elif 'verifier' in label_json.keys():
            return get_verifier_reward(query, prompt, label)

        # math 兜底
        else:
            raise NotImplementedError("尚未实现的rule-based reward")
    except Exception as e:
        return 0.0

def reward_func(queries, prompts, labels):
    rewards= []
    for query, prompt, label in zip(queries, prompts, labels):
        reward_ref = get_reward.remote((query, prompt, label))
        try:
            reward = ray.get(reward_ref, timeout=5)
        except ray.exceptions.GetTimeoutError:
            print("Math verify timeout error")
            ray.cancel(reward_ref, force=True)
            reward = 0.0
        rewards.append(reward)
    return torch.tensor(rewards)

# def reward_func(queries, prompts, labels):
#     rewards= []
#     for query, prompt, label in zip(queries, prompts, labels):
#         reward = get_reward((query, prompt, label))
#         rewards.append(reward)
#     return torch.tensor(rewards)

# def reward_func(queries, prompts, labels):
#     """
#     多进程版本的 reward_func，保证输出顺序与输入顺序一致
#     Args:
#         queries: 查询列表
#         prompts: 提示列表
#         labels: 标签列表
#         num_processes: 进程数，默认为None（自动选择）
#     Returns:
#         torch.Tensor: 奖励张量
#     """
#     # 准备参数元组列表
#     args_list = list(zip(queries, prompts, labels))
#     num_processes = 64
    
#     # 创建进程池
#     with Pool(processes=num_processes) as pool:
#         rewards = pool.map(_get_reward_wrapper, args_list)
#     return torch.tensor(rewards)

# def _get_reward_wrapper(args):
#     """
#     包装函数，用于处理单个任务的奖励计算和异常捕获
#     """
#     query, prompt, label = args
#     reward_ref = get_reward.remote((query, prompt, label))
#     try:
#         reward = ray.get(reward_ref, timeout=5)
#     except ray.exceptions.GetTimeoutError:
#         ray.cancel(reward_ref, force=True)
#         reward = 0.0
#     return reward

# def reward_func(queries, prompts, labels):
#     rewards = []
#     for query, prompt, label in zip(queries, prompts, labels):
#         try:
#             rewards.append(get_reward((query, prompt, label)))
#         except Exception as e:
#             print(f"Reward Error: {e}")
#             rewards.append(0.0)
#     return torch.tensor(rewards)

    # with ProcessPoolExecutor(max_workers=int(multiprocessing.cpu_count()//8)-4) as executor:
    #     items = [(query, prompt, label) for query, prompt, label in zip(queries, prompts, labels)]
    #     rewards = list(executor.map(get_reward, items))
    # return torch.tensor(rewards)



# 快速测试
if __name__ == "__main__":
    # queries = ['<role>HUMAN</role>Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.<role>ASSISTANT</role>Raymond III Count of Tripoli was a ruler who held the title of Prince of Antioch and count of Tripoli during the 13th century. Born around 1194, he was the son of Bohemond IV of Antioch and Constance of France. Raymond inherited the County of Tripoli after his father\'s death in 1205. In 1216, the County of Edessa was lost to the Muslims, and Raymond in 1217 became prince of Antioch through his marriage to Stephanie of Armenia the daughter of King Leo I of Armenia.\n\nIn *Early life and family*, Raymond was born in the County of Tripoli to Bohemond IV of Antioch and Constance of France. His father died in 1205, and Raymond succeeded him as Count of Tripoli. His mother was a member of the House of Courtenay, and her father was the French king Louis VIII. Raymond was a younger brother of Bohemond V of Antioch and was related to other European royal families.\n\nIn *Reign*, Raymond\'s reign was marked by conflicts with neighboring Muslim states and the struggle to maintain his territories. In 1216, the County of Edessa was lost to the Muslims, and Raymond in 1217 became prince of Antioch through his marriage to Stephanie of Armenia. In 1217, he was captured by the Muslims but was released after paying a large ransom. In 1233, Raymond was captured again by the Muslims and was only released after paying a large ransom. In 1236, Raymond went on a pilgrimage to Jerusalem and never returned to Antioch.\n\nIn *Death and legacy*, Raymond died in 1241 while on a pilgrimage to Jerusalem. He was succeeded by his son Bohemond VI of Antioch. Raymond\'s reign was marked by conflicts with neighboring Muslim states and the struggle to maintain his territories. He was married twice, first to Eschemia of Armenia and second to Stephanie of Armenia. He had several children, including Bohemond VI of Antioch, Conrad, and Isabella.'] * 100
    # prompts = ['<role>HUMAN</role>Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.<role>ASSISTANT</role>'] * 100
    # labels = ["{\"prompt\": \"Write a 300+ word summary of the wikipedia page \\\"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\\\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.\", \"key\": 1000, \"instruction_id_list\": [\"punctuation:no_comma\", \"detectable_format:number_highlighted_sections\", \"length_constraints:number_words\"], \"kwargs\": [{}, {\"num_highlights\": 3}, {\"relation\": \"at least\", \"num_words\": 300}]}"] * 100

    from tqdm import tqdm
    from transformers import AutoTokenizer
    # data = json.loads(open("/path/to/llm_pretrain/exp/397432/verify_debug/verify_debug/bad_case.jsonl").readlines()[-1])
    # print(data['response'])
    # queries = [data['prompt'] + data['response']]
    # prompts = [data['prompt']]
    # labels = [data['label']]   
    # rewards = reward_func(queries, prompts, labels)
    # print(rewards)
    # import time
    # for i in range(60):
    #     print(i)
    #     time.sleep(1)
    # time.sleep(60)
    # exit()



    wf = open("/path/to/llm_pretrain/exp/397432/verify_debug/verify_debug/bad_case_new.jsonl", 'w')
    all_rewards = []
    step=124
    tokenizer = AutoTokenizer.from_pretrained("/path/to/llm_pretrain/model_hub/Ling-Lite-0415", trust_remote_code=True)
    all_data = []
    for rank in tqdm(range(64)):
        data = [json.loads(line) for line in (open(f"/path/to/llm_pretrain/exp/397432/verify_debug/verify_debug/Rank-{rank}.jsonl").readlines())]
        for example in data:
            example['response'] = tokenizer.decode(example['response'])
        all_data.extend(data)
    for example in tqdm(all_data):
        wf.write(json.dumps(example) + "\n")
        wf.flush()
        
        queries = [example['prompt'] + example['response'] ]
        prompts = [example['prompt']]
        labels = [example['label']]
        rewards = reward_func(queries, prompts, labels)
        all_rewards.append(rewards.mean().item())
    
    print(sum(all_rewards) / len(all_rewards))