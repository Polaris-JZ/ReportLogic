import argparse
import json
import os
import hashlib
from collections import defaultdict


def parse_rl_rollout(root_directory, dataset_name):
    matching_files = [os.path.join(root_directory, name) for name in os.listdir(root_directory)]
    results = {}
    prompt_index = defaultdict(int)
    for path in matching_files:
        datalist = json.load(open(path, encoding='utf-8'))
        for data in datalist:
            result_data = {'dataset_name': dataset_name, 'category': 0}
            prompt = data['prompt']
            prompt_index[prompt] += 1
            prompt_str = prompt + str(prompt_index[prompt])
            mid = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()
            result_data['mid'] = mid
            result_data['prompt'] = prompt
            result_data['response'] = data['response']
            result_data['golden_answer'] = data['label']
            result_data['score'] = data['reward']
            results[mid] = result_data
    return results

def add_base_info(model_name, task_name, all_data):
    for key in all_data.keys():
        data = all_data[key]
        data['model_name'] = model_name
        data['task_name'] = task_name

        
def evaluate_rollout(args):
    results = parse_rl_rollout(os.path.join(args.exp_dir, f"evaluation_results/Step{args.step}/Max_new_tokens_8192"), f'Step{args.step}')
    add_base_info(f"{args.model_name}-step{args.step}", f'eval_rollout-step{args.step}', results)
    os.makedirs('./logs/case_logs/', exist_ok=True)
    with open(f'./logs/case_logs/{args.model_name}-step{args.step}-eval_rollout.jsonl', 'w', encoding='utf-8') as f:
        for key in results:
            data = results[key]
            f.write(json.dumps(data, ensure_ascii=False)+'\n')

def train_rollout(args):
    results = parse_rl_rollout(os.path.join(args.exp_dir, f"rollout_output/Step{args.step}"), f'Step{args.step}')
    add_base_info(args.model_name, 'train_rollout', results)
    os.makedirs('./logs/case_logs/', exist_ok=True)
    with open(f'./logs/case_logs/{args.model_name}-step{args.step}-train_rollout.jsonl', 'w', encoding='utf-8') as f:
        for key in results:
            data = results[key]
            f.write(json.dumps(data, ensure_ascii=False)+'\n')


def main(args):
    assert args.model_name != ""
    if args.output_type == 'train':
        train_rollout(args)
    elif args.output_type == 'eval':
        evaluate_rollout(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, and creates pytorch_model.bin, "
                    "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "--exp_dir",
        help="实验路径.",
    )
    parser.add_argument(
        '--output_type',
        type=str,
        choices=["train", 'eval'],
        default="train",
        help="训练 or 评测",
    )
    parser.add_argument(
        '--step',
        help="Step数"
    )
    parser.add_argument(
        '--model_name',
        help='模型名字，需要指定唯一'
    )
    args = parser.parse_args()
    main(args)

'''
Example usage:

python utils/convert_to_case_logs.py \
    --exp_dir /path/to/exp/my_run \
    --output_type train \
    --step 381 \
    --model_name my-model-v1

python utils/convert_to_case_logs.py \
    --exp_dir /path/to/exp/my_run \
    --output_type eval \
    --step 50 \
    --model_name my-model-v1
'''