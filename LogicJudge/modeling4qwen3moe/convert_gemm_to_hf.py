from huggingface_hub import save_torch_state_dict, load_state_dict_from_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import torch
from tqdm import tqdm
import json
import shutil
import os


def write_model(args):
    # Use load_state_dict_from_file to avoid requiring the Gemm environment
    # model = AutoModelForCausalLM.from_pretrained(args.input_dir, trust_remote_code=True, torch_dtype = torch.bfloat16)
    config = AutoConfig.from_pretrained(args.input_dir, trust_remote_code=True)
    files = os.listdir(args.input_dir)
    safttensor_files = [f"{args.input_dir}/{file_name}" for file_name in files if (file_name.endswith(".safetensors") and file_name.startswith("model"))]
    old_state_dict = {}
    for file_name in tqdm(safttensor_files):
        old_state_dict.update(load_state_dict_from_file(file_name, map_location="cuda"))

    new_state_dict = {}
    for key in old_state_dict.keys():
        if ".w1" in key:
            for i in range(config.num_experts):
                gate_proj_name = key.replace('w1', f"{i}.gate_proj.weight")
                gate_proj_weight = old_state_dict[key][i, :int(old_state_dict[key].size()[1]//2), :].cpu()
                new_state_dict[gate_proj_name] = gate_proj_weight

                up_proj_name = key.replace('w1', f"{i}.up_proj.weight")
                up_proj_weight = old_state_dict[key][i, int(old_state_dict[key].size()[1]//2):, :].cpu()
                new_state_dict[up_proj_name] = up_proj_weight
        elif "w2" in key:
            for i in range(config.num_experts):
                down_proj_name = key.replace('w2', f"{i}.down_proj.weight")
                down_proj_weight = old_state_dict[key][i, :, :].t().cpu()
                new_state_dict[down_proj_name] = down_proj_weight
        else:
          new_state_dict[key] = old_state_dict[key].cpu()
    save_torch_state_dict(new_state_dict, args.output_dir, max_shard_size="10GB")
    print(f"Model saved to {args.output_dir}")


def copy_file(src, tgt):
    if os.path.exists(src):
        shutil.copy(src, tgt)

def copy_files(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    copy_file(f"{args.input_dir}/configuration.json", f"{args.output_dir}/configuration.json")
    copy_file(f"{args.input_dir}/generation_config.json", f"{args.output_dir}/generation_config.json")
    copy_file(f"{args.input_dir}/merges.txt", f"{args.output_dir}/merges.txt")
    copy_file(f"{args.input_dir}/README.md", f"{args.output_dir}/README.md")
    copy_file(f"{args.input_dir}/special_tokens_map.json", f"{args.output_dir}/special_tokens_map.json")
    copy_file(f"{args.input_dir}/tokenizer.json", f"{args.output_dir}/tokenizer.json")
    copy_file(f"{args.input_dir}/tokenizer_config.json", f"{args.output_dir}/tokenizer_config.json")
    copy_file(f"{args.input_dir}/vocab.json", f"{args.output_dir}/vocab.json")
    copy_file(f"{args.input_dir}/chat_template.jinja", f"{args.output_dir}/chat_template.jinja")

    copy_file(f"{current_dir}/modeling_qwen3_moe_hf.py", f"{args.output_dir}/modeling_qwen3_moe.py")
    copy_file(f"{current_dir}/configuration_qwen3_moe.py", f"{args.output_dir}/configuration_qwen3_moe.py")


def add_config(args):
    config_json = json.loads(open(f"{args.input_dir}/config.json").read())
    del config_json['use_grouped_gemm_moe']
    del config_json['auto_map']
    with open(f"{args.output_dir}/config.json", 'w') as f:
        f.write(json.dumps(config_json, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, and creates pytorch_model.bin, "
                    "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of the Grouped-GEMM checkpoint to convert.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location of Output dir.",
    )
    parser.add_argument(
        "--memo",
        default="",
        help="Optional note appended to the run identifier.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    write_model(args)
    add_config(args)
    copy_files(args)
