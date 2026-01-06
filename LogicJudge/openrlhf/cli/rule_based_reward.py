import torch
from openrlhf.cli.ifeval_eval import get_ifeval_reward
from openrlhf.cli.serve_rm_math_boxed import get_correct_reward as get_boxed_reward
import json
import bailing_verifier.verifiers
from bailing_verifier.registry import VERIFIER_FACTORY
from bailing_verifier.verifiers.code.sandbox.ant_sandbox import ant_sandbox_verifier
from concurrent.futures import ProcessPoolExecutor
import multiprocessing



def get_verifier_reward(query, prompt, label):
    data_dict = json.loads(label)
    verification_func = VERIFIER_FACTORY.get(data_dict['verifier'])
    model_prediction = query.replace(prompt, "")
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

        elif 'verifier' in label_json.keys() and 'symbolic-match-verifier' in label_json['verifier']:
            return get_boxed_reward(query, prompt, label_json['gold']) 

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
        print(f"Error={e}")
        return 0.0

def reward_func(queries, prompts, labels):
    rewards = []
    for query, prompt, label in zip(queries, prompts, labels):
        try:
            rewards.append(get_reward((query, prompt, label)))
        except Exception as e:
            print(f"Reward Error: {e}")
            rewards.append(0.0)
    return torch.tensor(rewards)

    with ProcessPoolExecutor(max_workers=int(multiprocessing.cpu_count()//8)-4) as executor:
        items = [(query, prompt, label) for query, prompt, label in zip(queries, prompts, labels)]
        rewards = list(executor.map(get_reward, items))
    return torch.tensor(rewards)

# 快速测试
if __name__ == "__main__":
    # queries = ['<role>HUMAN</role>Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.<role>ASSISTANT</role>Raymond III Count of Tripoli was a ruler who held the title of Prince of Antioch and count of Tripoli during the 13th century. Born around 1194, he was the son of Bohemond IV of Antioch and Constance of France. Raymond inherited the County of Tripoli after his father\'s death in 1205. In 1216, the County of Edessa was lost to the Muslims, and Raymond in 1217 became prince of Antioch through his marriage to Stephanie of Armenia the daughter of King Leo I of Armenia.\n\nIn *Early life and family*, Raymond was born in the County of Tripoli to Bohemond IV of Antioch and Constance of France. His father died in 1205, and Raymond succeeded him as Count of Tripoli. His mother was a member of the House of Courtenay, and her father was the French king Louis VIII. Raymond was a younger brother of Bohemond V of Antioch and was related to other European royal families.\n\nIn *Reign*, Raymond\'s reign was marked by conflicts with neighboring Muslim states and the struggle to maintain his territories. In 1216, the County of Edessa was lost to the Muslims, and Raymond in 1217 became prince of Antioch through his marriage to Stephanie of Armenia. In 1217, he was captured by the Muslims but was released after paying a large ransom. In 1233, Raymond was captured again by the Muslims and was only released after paying a large ransom. In 1236, Raymond went on a pilgrimage to Jerusalem and never returned to Antioch.\n\nIn *Death and legacy*, Raymond died in 1241 while on a pilgrimage to Jerusalem. He was succeeded by his son Bohemond VI of Antioch. Raymond\'s reign was marked by conflicts with neighboring Muslim states and the struggle to maintain his territories. He was married twice, first to Eschemia of Armenia and second to Stephanie of Armenia. He had several children, including Bohemond VI of Antioch, Conrad, and Isabella.'] * 100
    # prompts = ['<role>HUMAN</role>Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.<role>ASSISTANT</role>'] * 100
    # labels = ["{\"prompt\": \"Write a 300+ word summary of the wikipedia page \\\"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\\\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.\", \"key\": 1000, \"instruction_id_list\": [\"punctuation:no_comma\", \"detectable_format:number_highlighted_sections\", \"length_constraints:number_words\"], \"kwargs\": [{}, {\"num_highlights\": 3}, {\"relation\": \"at least\", \"num_words\": 300}]}"] * 100

    from tqdm import tqdm
    all_rewards = []
    step=124

    data = json.loads(open(f"Step66.json").read())[:1024]
    queries = [example['prompt'] + example['response'] for example in data]
    prompts = [example['prompt'] for example in data]
    labels = [example['label'] for example in data]
    rewards = reward_func(queries, prompts, labels)
    raw_rewards = [example['reward'] for example in data]
    print(sum(rewards) / len(rewards))
    print(sum(raw_rewards) / len(raw_rewards))