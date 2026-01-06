from huggingface_hub import save_torch_state_dict, load_state_dict_from_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import torch
from tqdm import tqdm
import json
import shutil
import os
from aistudio_checkpoint.aistudio_base_checkpointer import lazy_load_checkpoint

def write_model(args):
    # 使用load_state_dict_from_file加载可以避免安装Gemm环境
    # model = AutoModelForCausalLM.from_pretrained(args.input_dir, trust_remote_code=True, torch_dtype = torch.bfloat16)
    config = AutoConfig.from_pretrained(args.input_dir, trust_remote_code=True)
    files = os.listdir(args.input_dir)
    safttensor_files = [f"{args.input_dir}/{file_name}" for file_name in files if (file_name.endswith(".safetensors") and file_name.startswith("model"))]
    old_state_dict = {}
    for file_name in tqdm(safttensor_files):
        old_state_dict.update(load_state_dict_from_file(file_name))

    new_state_dict = {}
    for key in old_state_dict.keys():
        if ".w1" in key:
            for i in range(config.num_experts):
                gate_proj_name = key.replace('w1', f"{i}.gate_proj.weight")
                gate_proj_weight = old_state_dict[key][i, :int(old_state_dict[key].size()[1]//2), :].clone()
                new_state_dict[gate_proj_name] = gate_proj_weight

                up_proj_name = key.replace('w1', f"{i}.up_proj.weight")
                up_proj_weight = old_state_dict[key][i, int(old_state_dict[key].size()[1]//2):, :].clone()
                new_state_dict[up_proj_name] = up_proj_weight
        elif "w2" in key:
            for i in range(config.num_experts):
                down_proj_name = key.replace('w2', f"{i}.down_proj.weight")
                down_proj_weight = old_state_dict[key][i, :, :].t().clone()
                new_state_dict[down_proj_name] = down_proj_weight
        else:
          new_state_dict[key] = old_state_dict[key]
    # 保存state_dict
    save_torch_state_dict(new_state_dict, args.output_dir, max_shard_size="10GB")

def copy_files(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(f"{args.input_dir}/special_tokens_map.json", os.path.join(args.output_dir, "special_tokens_map.json"))
    shutil.copy(f"{args.input_dir}/tokenizer.json", os.path.join(args.output_dir, "tokenizer.json"))
    shutil.copy(f"{args.input_dir}/tokenizer_config.json", os.path.join(args.output_dir, "tokenizer_config.json"))
    shutil.copy(f"{args.input_dir}/configuration_bailing_moe.py", os.path.join(args.output_dir, "configuration_bailing_moe.py"))
    shutil.copy(f"{current_dir}/modeling_bailing_moe_hf.py", os.path.join(args.output_dir, "modeling_bailing_moe.py"))
    if os.path.exists(f"{args.input_dir}/chat_template.jinja"):
        shutil.copy(f"{args.input_dir}/chat_template.jinja", os.path.join(args.output_dir, "chat_template.jinja"))


def add_config(args):
    config_json = json.loads(open(f"{args.input_dir}/config.json").read())
    del config_json['router_balance_loss_alpha']
    del config_json['router_z_loss_alpha']
    del config_json['use_grouped_gemm_moe']
    with open(f"{args.output_dir}/config.json", 'w') as f:
        f.write(json.dumps(config_json, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, and creates pytorch_model.bin, "
                    "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "--input_dir",
        help="Location of Bailing HF Checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        help="Location of Output dir.",
    )
    parser.add_argument(
        "--memo",
        default="",
        help="备注",
    )


    args = parser.parse_args()
    import re
    match = re.search(r'step-(\d+)', args.input_dir)
    if match:
        steps = match.group(1)
    else:
        steps = 0
    memo = args.memo + f"-step{steps}"

    if args.input_dir.startswith("aistudio"):
        args.input_dir = lazy_load_checkpoint(args.input_dir)

    ais_mnt_ckpter = None
    if not args.output_dir:
        from aistudio_checkpoint.aistudio_mnt_checkpointer import AistudioMntCheckpointer
        ais_mnt_ckpter = AistudioMntCheckpointer(is_merge_metadata=True)
        args.output_dir = ais_mnt_ckpter.get_save_dir(step=int(steps))

    if ais_mnt_ckpter is None:
        os.makedirs(args.output_dir, exist_ok=True)
        
    write_model(args)
    add_config(args)
    copy_files(args)

    if ais_mnt_ckpter is not None:
        ais_mnt_ckpter.commit(memo=memo)
