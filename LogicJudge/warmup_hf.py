import argparse
from transformers import AutoModelForCausalLM
from aistudio_checkpoint.aistudio_base_checkpointer import lazy_load_checkpoint
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pretrain', type=str, default='')
    args = parser.parse_args()
    pretrain = args.pretrain
    if isinstance(pretrain, str) and pretrain.startswith("aistudio"):
        pretrain = lazy_load_checkpoint(resuming_path=pretrain)
    model = AutoModelForCausalLM.from_pretrained(pretrain, trust_remote_code=True, torch_dtype = torch.bfloat16)
