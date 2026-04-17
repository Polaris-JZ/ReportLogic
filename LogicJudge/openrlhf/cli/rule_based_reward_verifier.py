import torch
from openrlhf.cli.ifeval_eval import get_ifeval_reward
from openrlhf.cli.serve_rm_math_boxed import get_correct_reward as get_boxed_reward
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import ray
from multiprocessing import Pool

if not ray.is_initialized():
    ray.init()


def aggregation_rm_func(rewards, query, prompt, label):
    return rewards[0] * rewards[1]


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
            return get_ifeval_reward(query, prompt, label)

        else:
            raise NotImplementedError("No rule-based reward implemented for this label schema.")
    except Exception as e:
        return 0.0

def reward_func(queries, prompts, labels):
    rewards = []
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


if __name__ == "__main__":
    import os
    from tqdm import tqdm
    from transformers import AutoTokenizer

    debug_dir = os.environ.get("VERIFY_DEBUG_DIR", "/path/to/verify_debug")
    tokenizer_path = os.environ.get("CHAT_TOKENIZER_PATH", "/path/to/checkpoint/chat-tokenizer")

    wf = open(os.path.join(debug_dir, "bad_case_new.jsonl"), 'w')
    all_rewards = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    all_data = []
    for rank in tqdm(range(64)):
        data = [json.loads(line) for line in open(os.path.join(debug_dir, f"Rank-{rank}.jsonl")).readlines()]
        for example in data:
            example['response'] = tokenizer.decode(example['response'])
        all_data.extend(data)
    for example in tqdm(all_data):
        wf.write(json.dumps(example) + "\n")
        wf.flush()

        queries = [example['prompt'] + example['response']]
        prompts = [example['prompt']]
        labels = [example['label']]
        rewards = reward_func(queries, prompts, labels)
        all_rewards.append(rewards.mean().item())

    print(sum(all_rewards) / len(all_rewards))
