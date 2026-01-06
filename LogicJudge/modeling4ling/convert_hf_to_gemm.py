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
    # model = AutoModelForCausalLM.from_pretrained(args.input_dir, trust_remote_code=True, torch_dtype = torch.bfloat16)
    config = AutoConfig.from_pretrained(args.input_dir, trust_remote_code=True)
    files = os.listdir(args.input_dir)
    safttensor_files = [f"{args.input_dir}/{file_name}" for file_name in files if (file_name.endswith(".safetensors") and file_name.startswith("model"))]
    old_state_dict = {}
    for file_name in tqdm(safttensor_files):
        old_state_dict.update(load_state_dict_from_file(file_name))

    # old_state_dict = model.state_dict()
    new_state_dict = {}
    for key in old_state_dict.keys():
        if 'mlp.experts' in key:
          continue
        new_state_dict[key] = old_state_dict[key]

    for layer_index in tqdm(range(config.num_hidden_layers), desc="merge experts"):
        v1 = torch.stack([old_state_dict[f"model.layers.{layer_index}.mlp.experts.{i}.gate_proj.weight"] for i in range(config.num_experts)])
        # v1 = torch.stack([model.model.layers[layer_index].mlp.experts[i].gate_proj.weight for i in range(len(model.model.layers[layer_index].mlp.experts))])
        w1 = torch.stack([old_state_dict[f"model.layers.{layer_index}.mlp.experts.{i}.up_proj.weight"] for i in range(config.num_experts)])
        # w1 = torch.stack([model.model.layers[layer_index].mlp.experts[i].up_proj.weight for i in range(len(model.model.layers[layer_index].mlp.experts))])
        w1 = torch.cat([v1, w1], dim=1)
        w2 = torch.stack([old_state_dict[f"model.layers.{layer_index}.mlp.experts.{i}.down_proj.weight"].t() for i in range(config.num_experts)])
        # w2 = torch.stack([model.model.layers[layer_index].mlp.experts[i].down_proj.weight.t() for i in range(len(model.model.layers[layer_index].mlp.experts))])
        new_state_dict[f"model.layers.{layer_index}.mlp.experts.w1"] = w1.to(torch.bfloat16)
        new_state_dict[f"model.layers.{layer_index}.mlp.experts.w2"] = w2.to(torch.bfloat16)
    # 保存state_dict
    
    if config.norm_head and 'lm_head.weight' in new_state_dict:
        import torch.nn.functional as F
        print("Delete norm head in config!!!!")
        print(f"{new_state_dict['lm_head.weight']=}")
        new_state_dict["lm_head.weight"] = F.normalize(new_state_dict["lm_head.weight"], dim=0, p=2, eps=1e-7)
        print(f"{new_state_dict['lm_head.weight']=}")
    
    save_torch_state_dict(new_state_dict, args.output_dir, max_shard_size="10GB")

def copy_files(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(f"{args.input_dir}/special_tokens_map.json", os.path.join(args.output_dir, "special_tokens_map.json"))
    shutil.copy(f"{args.input_dir}/tokenizer.json", os.path.join(args.output_dir, "tokenizer.json"))
    shutil.copy(f"{args.input_dir}/tokenizer_config.json", os.path.join(args.output_dir, "tokenizer_config.json"))
    shutil.copy(f"{args.input_dir}/configuration_bailing_moe.py", os.path.join(args.output_dir, "configuration_bailing_moe.py"))
    shutil.copy(f"{current_dir}/modeling_bailing_moe_gemm.py", os.path.join(args.output_dir, "modeling_bailing_moe.py"))
    if os.path.exists(f"{args.input_dir}/chat_template.jinja"):
        shutil.copy(f"{args.input_dir}/chat_template.jinja", os.path.join(args.output_dir, "chat_template.jinja"))



def add_config(args):
    config_json = json.loads(open(f"{args.input_dir}/config.json").read())
    config_json['router_balance_loss_alpha'] = 0.015
    config_json['router_z_loss_alpha'] = 0.0001
    config_json['use_grouped_gemm_moe'] = True
    config_json['norm_head'] = False
    if 'architectures' in config_json and 'BailingMoeForRewardModel' in config_json['architectures']:
        config_json['auto_map']['AutoModelForSequenceClassification'] = "modeling_bailing_moe.BailingMoeForRewardModel"
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

    if args.input_dir.startswith("aistudio"):
        args.input_dir = lazy_load_checkpoint(args.input_dir)

    ais_mnt_ckpter = None
    if not args.output_dir:
        from aistudio_checkpoint.aistudio_mnt_checkpointer import AistudioMntCheckpointer
        ais_mnt_ckpter = AistudioMntCheckpointer(is_merge_metadata=True)
        args.output_dir = ais_mnt_ckpter.get_save_dir(step=0)

    if ais_mnt_ckpter is None:
        os.makedirs(args.output_dir, exist_ok=True)
        
    os.makedirs(args.output_dir, exist_ok=True)
    write_model(args)
    add_config(args)
    copy_files(args)

    if ais_mnt_ckpter is not None:
        ais_mnt_ckpter.commit(memo=args.memo)

