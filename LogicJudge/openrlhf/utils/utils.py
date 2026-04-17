import os
from typing import List

from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer
import torch
import ray


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(pretrain, model=None, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        # elif os.path.isdir(dataset):
        #     data = load_from_disk(dataset)
        #     print(f"loaded {dataset} from disk")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def print_gpu_memory_details():
    # 检查设备是否可用
    if torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info()
        free_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        print(f"Total: {total_memory:.2f} GB; Free: {free_memory:.2f} GB")
    else:
        print("CUDA is not available.")


def get_ckpt_path(args, model_type):
    return os.path.join(args.ckpt_path, model_type)

    
# 通过路径动态寻找文件加载自定义的奖励函数
def load_func_from_path(path, func_name):
    print(f"Loading custom `{func_name}` from {path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location(func_name, path)
    func_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(func_module)
    return getattr(func_module, func_name)

def sleep_vllm(vllm_engines, flag=True):
    if flag:
        refs = []
        for engine in vllm_engines:
            refs.append(engine.sleep.remote())
        ray.get(refs, timeout=5)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

def wakeup_vllm(vllm_engines, flag=True):
    if flag:
        print("Before wakeup vllm:")
        print_gpu_memory_details()
        refs = []
        for engine in vllm_engines:
            refs.append(engine.wake_up.remote())
        ray.get(refs, timeout=5)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

# 打乱原始顺序
def distributed_order(world_size, data):
    return sum([data[i::world_size] for i in range(world_size)], [])

# 恢复原始顺序
def recover_original_order(world_size, interval_data):
    # 计算每个块的长度
    chunk_size = len(interval_data) // world_size
    # 将打乱的数据拆分为多个块
    chunks = [interval_data[i * chunk_size: (i + 1) * chunk_size] for i in range(world_size)]
    # 恢复原始顺序
    original_data = []
    for i in range(chunk_size):
        for j in range(world_size):
            original_data.append(chunks[j][i])
    return original_data

def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)
