import ray
import re
import os
import json
from glob import glob
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

"""
Read JSONL data from input_file_path, extract the content of the specified key as the question,
run inference with the model at model_path (tp = tensor parallel size; 1 on H200, 2 on A100/H800),
and save results to output_file_path. The output extends each input record with a "judge_results"
list of dicts, each containing {"response", "response_length", "judge_model"}.
"""

input_files = glob('/path/to/data/testset/benchmarks/*/*.jsonl')
output_dir = '/path/to/exp/sft_wocl_ckpts/eval'
input_key = "input"

model_path = "/path/to/checkpoint/sft_wocl"
tp = 1  # tp=1 on H200, tp=2 on H800/A100 is usually faster


ray.init()

CHOICES = ['A>B', 'A=B', 'A<B']

def check_output_format(text):
    analysis = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    if not (len(analysis) == 1 and bool(analysis[0].strip())):
        return None

    # pred = re.findall(r'</think>\n\n(.*?)<\|im_end\|>', text, re.DOTALL)
    pred = re.search(r'</think>\n\n(.{3})$', text, re.DOTALL)
    if pred and pred.group(1) in CHOICES:
        return pred.group(1)
    else:
        return None

def get_reward(output, label):
    valid_output = check_output_format(output)
    if valid_output is None:
        return -1.0
    
    if valid_output == label:
        return 1.0
    else:
        return 0.0


def get_local_path(model_path):
    return model_path


def read_jsonl(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
def write_jsonl(objs, data_path):
    with open(data_path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

@ray.remote
class InferEngine:
    def __init__(self, model_path: str, tp: int, engine_idx: int):
        self.llm = LLM(
            model=get_local_path(model_path),
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=1.0,
            top_k=-1,
            max_tokens=6000,
        )
        self.engine_idx = engine_idx
        
    def generate(self, prompts: list[str]) -> tuple[list[str], list[int]]:
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=self.engine_idx == 0)
        response_texts = [output.outputs[0].text for output in outputs]
        response_lengths = [len(output.outputs[0].token_ids) for output in outputs]
        return response_texts, response_lengths


def main():
    n_gpus = int(ray.cluster_resources().get("GPU", 0))
    assert (n_gpus >= tp) and (n_gpus % tp == 0), "n_gpus必须是tp的倍数"
    n_engines = n_gpus // tp

    tokenizer = AutoTokenizer.from_pretrained(get_local_path(model_path))
    engines = [InferEngine.options(num_gpus=tp).remote(model_path, tp, i) for i in range(n_engines)]
    print(f"创建 {n_engines} 个推理引擎, 每个使用 {tp} 个GPU")

    start = 0
    prompts, total_data_list = [], []
    indexes = dict()
    for input_file_path in input_files:
        data_list = read_jsonl(input_file_path)
        total_data_list += data_list
        testset_name = os.path.basename(input_file_path).replace('.jsonl', '')
        total = len(data_list)
        for item in data_list:
            messages = [{"role": "user", "content": item[input_key]}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        end = start + total
        indexes[testset_name] = (start, end)
        start = end
        print(f"File: {input_file_path} 读取 {len(data_list)} 个数据样本")
    print(f"共读取 {len(prompts)} 个数据样本")

    # 分批数据到每个引擎,余数为i的数据分配到i号引擎
    chunks = [
        [p for j, p in enumerate(prompts) if j % n_engines == i]
        for i in range(n_engines)
    ]

    result_refs = [engines[i].generate.remote(chunk) for i, chunk in enumerate(chunks)]
    chunk_results = ray.get(result_refs)

    # 推理结果合并&保存
    for i in range(n_engines):
        rsp_texts, rsp_lengths = chunk_results[i]
        for j in range(len(rsp_texts)):
            idx = j * n_engines + i
            # if "judge_results" not in data_list[idx]:
            #     data_list[idx]["judge_results"] = []
            # data_list[idx]["judge_results"].append({
            #     "response": rsp_texts[j],
            #     "response_length": rsp_lengths[j],
            #     "judge_model": model_path,
            # })
            total_data_list[idx]['response'] = rsp_texts[j]
            total_data_list[idx]['response_length'] = rsp_lengths[j]
            total_data_list[idx]['judge_model'] = model_path
            total_data_list[idx]['reward'] = get_reward(rsp_texts[j], total_data_list[idx]['label'])

    # write_jsonl(data_list, output_file_path)
    model_name = os.path.basename(model_path)
    os.makedirs(output_dir, exist_ok=True)
    for testset, (start, end) in indexes.items():
        output_file_path = os.path.join(output_dir, f"{testset}_{model_name}.jsonl")
        with open(output_file_path, 'w') as fout:
            json.dump(total_data_list[start: end], fout, ensure_ascii=False, indent=2)

    for engine in engines:
        ray.kill(engine)


if __name__ == "__main__":
    main()