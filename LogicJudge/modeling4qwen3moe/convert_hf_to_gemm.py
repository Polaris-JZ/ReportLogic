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
        old_state_dict.update(load_state_dict_from_file(file_name, map_location="cuda"))

    # old_state_dict = model.state_dict()
    new_state_dict = {}
    for key in old_state_dict.keys():
        if 'mlp.experts' in key:
          continue
        new_state_dict[key] = old_state_dict[key].cpu()

    for layer_index in tqdm(range(config.num_hidden_layers), desc="merge experts"):
        v1 = torch.stack([old_state_dict[f"model.layers.{layer_index}.mlp.experts.{i}.gate_proj.weight"] for i in range(config.num_experts)])
        # v1 = torch.stack([model.model.layers[layer_index].mlp.experts[i].gate_proj.weight for i in range(len(model.model.layers[layer_index].mlp.experts))])
        w1 = torch.stack([old_state_dict[f"model.layers.{layer_index}.mlp.experts.{i}.up_proj.weight"] for i in range(config.num_experts)])
        # w1 = torch.stack([model.model.layers[layer_index].mlp.experts[i].up_proj.weight for i in range(len(model.model.layers[layer_index].mlp.experts))])
        w1 = torch.cat([v1, w1], dim=1)
        w2 = torch.stack([old_state_dict[f"model.layers.{layer_index}.mlp.experts.{i}.down_proj.weight"].t() for i in range(config.num_experts)])
        # w2 = torch.stack([model.model.layers[layer_index].mlp.experts[i].down_proj.weight.t() for i in range(len(model.model.layers[layer_index].mlp.experts))])
        new_state_dict[f"model.layers.{layer_index}.mlp.experts.w1"] = w1.cpu()
        new_state_dict[f"model.layers.{layer_index}.mlp.experts.w2"] = w2.cpu()
    # 保存state_dict
    
    if getattr(config, "norm_head", False) and 'lm_head.weight' in new_state_dict:
        import torch.nn.functional as F
        print("Delete norm head in config!!!!")
        print(f"{new_state_dict['lm_head.weight']=}")
        new_state_dict["lm_head.weight"] = F.normalize(new_state_dict["lm_head.weight"], dim=0, p=2, eps=1e-7).cpu()
        print(f"{new_state_dict['lm_head.weight']=}")
    
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

    copy_file(f"{current_dir}/modeling_qwen3_moe_gemm.py", f"{args.output_dir}/modeling_qwen3_moe.py")
    copy_file(f"{current_dir}/configuration_qwen3_moe.py", f"{args.output_dir}/configuration_qwen3_moe.py")


def add_config(args):
    config_json = json.loads(open(f"{args.input_dir}/config.json").read())
    config_json['use_grouped_gemm_moe'] = True
    config_json['auto_map'] = {
        "AutoConfig": "configuration_qwen3_moe.Qwen3MoeConfig",
        "AutoModel": "modeling_qwen3_moe.Qwen3MoeModel",
        "AutoModelForCausalLM": "modeling_qwen3_moe.Qwen3MoeForCausalLM",
    }
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

