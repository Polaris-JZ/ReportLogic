#-*- coding:utf-8
import os
import argparse
from pathlib import Path
from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf, KMConf
from pypai.job import RLJobBuilder
from pypai.conf import ExecConf, KMConf
from pypai.conf.retry_strategy import RetryStrategy, RetryPolicy
from aistudio_common.openapi.models.data_store import DataStore

p_yuyan_mdata_cpfs = DataStore(mount_point="/mdata_p_x_01", store_name="mdata-p-x-01", sub_path="")
# OSS_CODE_PATH = f"chilin.fcl/zuoyi/code"
OSS_CODE_PATH = "xiaoqi.lh/code"
MALAY_KEY = "-i LTAI5tEj6JNf9Kd3etqwVjEL -k h4IKUz93kBjgykvmzvII3nrw1fE4O4"
HUANAN_KEY = "-i LTAI5tJzf6E6RcfvAbdDe5ZS -k lQzCqhrp1YCOl3CGke195pAfmlzE0C"
NAS_CMD = """
mkdir /mnt/nas && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-39-elt86.cn-heyuan-alipay.nas.aliyuncs.com:/ /mnt/nas
mkdir /mnt/bailing1  && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-10-emx24.cn-heyuan-alipay.nas.aliyuncs.com:/ /mnt/bailing1
mkdir /agent && mount -t nfs -o vers=3,nolock,proto=tcp alipay-heyuan-33-ssy91.cn-heyuan-alipay.nas.aliyuncs.com:/ /agent
if [ ! -d /adabench_nas ]; then mkdir /adabench_nas; fi; mount -t nfs -o vers=3,nolock,proto=tcp alipayshnas-026-jxd68.cn-shanghai-eu13-a01.nas.aliyuncs.com:/ /adabench_nas
mkdir /ad
mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-39-elt86.cn-heyuan-alipay.nas.aliyuncs.com:/ /ad
cd /path/to/data/SFT/LLaMA-Factory-upload
"""

def get_nltk_data(args):
    if args.app in ['gbank', 'sgtest']:
        return "/path/to/nltk_data"
    elif 'huanan' in args.app:
        return "/input/chilin.fcl/data/lite_rl_data/eval_data/nltk_data"
    elif args.app in ['yuyangpudev', 'malaysia', "malaysiadev", "malaysia2", "llmdebugmalaih800", ]:
        return "/input/chilin.fcl/data/lite_rl_data/eval_data/nltk_data"
    elif args.app in ['lings3', 'lings3temp', 'lings3debug', 'lings3eval']:
        return "/input/chilin.fcl/data/lite_rl_data/eval_data/nltk_data"
    elif args.app in ['lingmlc', 'lingmlcdebug', 'lingmlctemp1']:
        return "/input/sunxiaqing.sxq/post_training/data/nltk_data"
    else:
        raise NotImplementedError

def get_reward_pretrain(config_path):
    for line in open(config_path).readlines():
        if '--reward_pretrain' in line:
            return line.replace("--reward_pretrain", "").replace("\\", "").strip()
    return None

def get_pretrain(config_path):
    for line in open(config_path).readlines():
        if '--pretrain' in line:
            return line.replace("--pretrain", "").replace("\\", "").strip()

def get_exp_path(config_path):
    for line in open(config_path).readlines():
        if '--save_path' in line:
            return line.replace("--save_path", "").replace("\\", "").strip().rstrip('\\')

def get_command(args):
    with open(args.config_path) as rf:
        return rf.read().replace("\\", "").replace("\n", "")

def get_init_command(args):
    exp_name = Path(args.exp_path).name

    init_command = """
pip install grouped-gemm==0.1.4
pip install megablocks==0.9.0
pip install transformers==4.57.1
pip install alipay-pcache==0.1.8
pip install unifile-sdk==0.1.19
pip install aistudio-checkpoint -U
export USERNUMBER="496380"
"""
    if args.app in ["malaysia", "yuyangpudev", "malaysiadev", "malaysia2", "llmdebugmalaih800", "lingmlc", "lingmlcdebug", "lingmlctemp1"]:
        code_init_command = f"""mkdir -p {args.exp_path}
ossutil64 config -e oss-ap-southeast-1-internal.aliyuncs.com {MALAY_KEY}
ossutil64 cp -f oss://gpu-wulanchabu-znjh-os30/{OSS_CODE_PATH}/{exp_name}/OpenRLHF.tar.gz {args.exp_path}/OpenRLHF.tar.gz
cd {args.exp_path}
tar -zxvf OpenRLHF.tar.gz
rm -f OpenRLHF.tar.gz
ls -lh {args.exp_path}/OpenRLHF
"""  # 解压代码库
    elif 'huanan' in args.app or args.app in ['lings3', 'lings3temp', 'lings3debug', 'lings3eval']:
        code_init_command = f"""mkdir -p {args.exp_path}
# ossutil64 config -e https://oss-cn-shenzhen.aliyuncs.com {HUANAN_KEY}
# ossutil64 cp -f oss://antsys-gpu-shenzhen-code/{OSS_CODE_PATH}/{exp_name}/OpenRLHF.tar.gz {args.exp_path}/OpenRLHF.tar.gz
cd {args.exp_path}
# tar -zxvf OpenRLHF.tar.gz
# rm -f OpenRLHF.tar.gz
ls -lh {args.exp_path}/OpenRLHF
"""
    elif args.app in ['gbank']:
        code_init_command = NAS_CMD
    else:
        raise NotImplementedError

    code_init_command += f"""
ls {args.exp_path}/OpenRLHF/vllm4ling -lh 
cp {args.exp_path}/OpenRLHF/vllm4ling/models/bailing_moe.py /opt/conda/lib/python3.10/site-packages/vllm/model_executor/models
cp {args.exp_path}/OpenRLHF/vllm4ling/models/registry.py /opt/conda/lib/python3.10/site-packages/vllm/model_executor/models
cp {args.exp_path}/OpenRLHF/vllm4ling/configs/__init__.py /opt/conda/lib/python3.10/site-packages/vllm/transformers_utils/configs
cp {args.exp_path}/OpenRLHF/vllm4ling/configs/bailing_moe.py /opt/conda/lib/python3.10/site-packages/vllm/transformers_utils/configs
cp {args.exp_path}/OpenRLHF/dockerfile/torch_checkpoint_engine.py /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py 
cp {args.exp_path}/OpenRLHF/dockerfile/grouped_gemm_moe.py  /opt/conda/lib/python3.10/site-packages/atorch/modules/moe/grouped_gemm_moe.py
cp {args.exp_path}/OpenRLHF/modeling4qwen3moe/qwen3_moe.py  /opt/conda/lib/python3.10/site-packages/vllm/model_executor/models/qwen3_moe.py
python {args.exp_path}/OpenRLHF/warmup_hf.py --pretrain {get_pretrain(args.config_path)}
"""
    if get_reward_pretrain(args.config_path) is not None:
        code_init_command += f"\npython {args.exp_path}/OpenRLHF/warmup_hf.py --pretrain {get_reward_pretrain(args.config_path)}"

    init_command += code_init_command
    return init_command

def upload_code(args):
    exp_name = Path(args.exp_path).name

    commands = ""
    if args.app in ["malaysia", "yuyangpudev", "malaysiadev", "malaysia2", "llmdebugmalaih800", "malaysiareasoneval"] or args.app in ['lingmlctemp1', 'lingmlc', 'lingmlcdebug', 'lingmlceval']:
        if "33.151.11.248" not in open("/etc/hosts", 'r').read():
            with open("/etc/hosts", "a") as f:
                f.write("\n33.151.11.248 gpu-wulanchabu-znjh-os30.oss-ap-southeast-1.aliyuncs.com")
        commands += f"cd .. && tar -zcvf OpenRLHF.tar.gz OpenRLHF --exclude='.git' --exclude='__pycache__' && ossutil64 config -e oss-ap-southeast-1.aliyuncs.com {MALAY_KEY} && ossutil64 cp -f OpenRLHF.tar.gz oss://gpu-wulanchabu-znjh-os30/{OSS_CODE_PATH}/{exp_name}/OpenRLHF.tar.gz && rm -f OpenRLHF.tar.gz"  # 压缩代码库加速上传
    elif 'huanan' in args.app or args.app in ['lings3', 'lings3temp', 'lings3debug', 'lings3eval']:
        commands += f"cd .. && tar -zcvf OpenRLHF.tar.gz OpenRLHF --exclude='.git' --exclude='__pycache__' && ossutil64 config -e https://oss-cn-shenzhen.aliyuncs.com {HUANAN_KEY} && ossutil64 cp -f OpenRLHF.tar.gz oss://antsys-gpu-shenzhen-code/{OSS_CODE_PATH}/{exp_name}/OpenRLHF.tar.gz && rm -f OpenRLHF.tar.gz"
    elif args.app in ['gbank']:
        commands = f"cd .. && mkdir -p {args.exp_path} && cp -rf OpenRLHF {args.exp_path}/ && ls {args.exp_path}/OpenRLHF"

    os.system(commands)


def set_os(args):
    if args.app in ['yuyanhuanan', 'yuyanhuananeval', 'llmdebughuananh800']:
        os.environ["IDE_COMMON_OSS_BUCKET"] = "antsys-gpu-shenzhen-code"
        os.environ["ENV_AISTUDIO_HOST"] = "aistudio.alipay.com"
        os.environ["AISTUDIO_SITE_ENUM"] = "HUANAN"
    elif args.app in ['yuyangpudev', 'malaysia', "malaysiadev", "malaysia2", "llmdebugmalaih800", "lingmlc", "lingmlcdebug", "lingmlctemp1"]:
        os.environ["ENV_AISTUDIO_HOST"] = "aistudio.alipay.com"
        os.environ["IDE_COMMON_OSS_BUCKET"] = "gpu-sg-aistudiocode"
        os.environ["AISTUDIO_SITE_ENUM"] = "SINGAPORE"
    elif args.app in ['gbank', 'sgtest']:
        pass
    elif args.app in ['lings3', 'lings3temp', 'lings3debug', 'lings3eval']:
        pass
    else:
        raise NotImplementedError


def get_data_stores(args):
    if args.app in ['yuyanhuanan', 'yuyanhuananeval', 'llmdebughuananh800']:
        data_stores = [
            DataStore(
                mount_point="/input",  # 挂载到当前环境的路径
                store_name="huanan-yuyan-cpfs",  # nas 盘名字
                sub_path="",  # 所要挂载的 nas 盘的路径
            ),
            DataStore(
                mount_point="/mdata_huanan",
                store_name="mdata-huanan",
                sub_path="",
            ),
            DataStore(
                mount_point="/mnt/ais-hn/dataset",
                store_name="ais-pcache-huanan-dataset",
                sub_path="/mnt/ais-hn/dataset",
            ),
            DataStore(
                mount_point="/mnt/pcachetest/checkpoint",    
                store_name="ais-pcache-huanan-checkpoint",           
                sub_path="/mnt/pcachetest/checkpoint"                     
            ),
        ]
    elif args.app in ['yuyangpudev', 'malaysia', "malaysiadev", "malaysia2", "llmdebugmalaih800"]:
        data_stores = [
            DataStore(
                mount_point="/input",  # 挂载到当前环境的路径
                store_name="malaysia-bmcpfs",  # nas 盘名字
                sub_path="",  # 所要挂载的 nas 盘的路径
            ),
            DataStore(
                mount_point="/aistudio/checkpoint",  # 挂载到当前环境的路径
                store_name="aistudio-checkpoint-malaysia-01",  # nas 盘名字
                sub_path="",  # 所要挂载的 nas 盘的路径
            ),
        ]
    elif args.app in ['sgtest']:
        data_stores = [
            DataStore(mount_point="/mnt/antsys-dlrover-sg-data", store_name="sg52-nlp-antsys-dlrover-sg-data", sub_path="/mnt/antsys-dlrover-sg-data"),
            DataStore(mount_point="/mnt/ais-aidc-sg52/dataset", store_name="ais-pcache-ais-aidc-sg52-dataset", sub_path="/mnt/ais-aidc-sg52/dataset"),
            DataStore(mount_point="/mnt/antsys-dlrover-sg-ckpt", store_name="sg52-nlp-antsys-dlrover-sg-ckpt", sub_path="/mnt/antsys-dlrover-sg-ckpt"),
            DataStore(mount_point="/mnt/antsys-dlrover-sg-h100", store_name="sg52-nlp-antsys-dlrover-sg-h100", sub_path="/mnt/antsys-dlrover-sg-h100"),
            DataStore(mount_point='/mnt/antsys-dlrover-sg-aoe', store_name="sg52-nlp-antsys-dlrover-sg-aoe", sub_path="")
        ]
    elif args.app in ['lings3', 'lings3temp', 'lings3debug', 'lings3eval']:
        data_stores = [
            DataStore(
                mount_point="/input",  # 挂载到当前环境的路径
                store_name="s3-yuyan-cpfs",  # nas 盘名字
                sub_path="",  # 所要挂载的 nas 盘的路径
            ),
        ]
    elif args.app in ['lingmlc', 'lingmlcdebug', "lingmlctemp1"]:
        data_stores = [
            DataStore(
                mount_point="/input",  # 挂载到当前环境的路径
                store_name="mlc-yuyan-cpfs",  # nas 盘名字
                sub_path="",  # 所要挂载的 nas 盘的路径
            ),
        ]
    elif args.app in ['gbank']:
        data_stores = None
    else:
        raise NotImplementedError
    return data_stores

def get_image(args):
    if 'huanan' in args.app:
        return "acr-sz-ant-registry-vpc.cn-shenzhen.cr.aliyuncs.com/aii/aistudio:aistudio-189587869-1653342929-1751958222048"
    elif args.app in ['yuyangpudev', 'malaysia', "malaysiadev", "malaysia2", "llmdebugmalaih800", "lingmlc", "lingmlcdebug", "lingmlctemp1"]:
        return "acr-singapore-registry-vpc.ap-southeast-1.cr.aliyuncs.com/aii/aistudio:aistudio-189587869-1653342929-1751958222048"
    elif args.app in ['gbank', 'sgtest']:
        return "reg.docker.alibaba-inc.com/aii/aistudio:aistudio-189587869-1653342929-1751958222048"
        # raise NotImplementedError("未为此集群配置环境")
    elif args.app in ['lings3', 'lings3temp', 'lings3debug', 'lings3eval']:
        return "acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/aii/aistudio:aistudio-189587869-1653342929-1751958222048"
    else:
        raise NotImplementedError

def modify_config_yaml(path, modify_key, modify_value):
    from ruamel.yaml import YAML
    yaml = YAML()
    # 打开yaml文件
    with open(path, 'r') as f:
        data = yaml.load(f)

    # 修改dir的值
    data[modify_key] = modify_value

    # 将修改后的数据写回到yaml文件
    with open(path, 'w') as f:
        yaml.dump(data, f)

    print(f"File {path} has been written.")


def main(args):
    set_os(args)
    data_stores = get_data_stores(args)
    # working_dir = f"{args.exp_path}/OpenRLHF"
    if args.app in ['gbank']:
        working_dir = f"{args.exp_path}/OpenRLHF"
    else:
        working_dir = f"/root"
    app = args.app
    use_code_in_nas = True if (args.app in ['sgtest', 'gbank']) else False
    num_node = args.num_node
    node_gpu_num = args.gpu_num
    config = args.config_path
    exp_path = args.exp_path
    gpu_type = args.gpu_type
    # modify_config_yaml(config, "output_dir", exp_path)
    
    if args.upload_code:
        upload_code(args)
        
    command = get_command(args)
    init_command = get_init_command(args)
    

    print("Command:\n", command)
    # exit()
    print("Init_command:\n", init_command)

    print(f"k8s app name:{app}")
    print(f"num_node:{num_node}")
    print(f"node_gpu_num:{node_gpu_num}")
    print(f"config file path:{config}")
    print(f"exp_path:{exp_path}")
    print(f"use_code_in_nas:{use_code_in_nas}")
    print(f"k8s gpu priority:{args.priority}")

    # num_node = 1
    # node_gpu_num = 8
    master = ExecConf(cpu=94, memory=1572864, disk_m=2021070, gpu_num=node_gpu_num, num=1, gpu_type=gpu_type)
    worker = None
    if num_node > 1:
        worker = ExecConf(
            cpu=94,
            memory=1572864,
            disk_m=2021070,
            gpu_num=node_gpu_num,
            num=num_node-1,
            gpu_type=gpu_type,
        )

    envs = {
        # "PYTHONPATH": f"/input/tangzuoli.tzl/Code4RL/bailing_verifier:{args.exp_path}/OpenRLHF:/root:$PYTHONPATH",
        "PYTHONPATH": f"/input/tangzuoli.tzl/Code4RL/bailing_verifier:{args.exp_path}/OpenRLHF:$PYTHONPATH",
        "VLLM_HOST_IP": "0.0.0.0",
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.9",
        "LD_LIBRARY_PATH": "/opt/conda/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH",
        'MOE_MLP_PREFIX': 'False',
        "NLTK_DATA": get_nltk_data(args)
    }
    image=get_image(args)
    rs = RetryStrategy(retry_policy=RetryPolicy.ON_FAILURE, max_attempt=10)
    assert args.app != "sgtest", "SG52集群不支持Ray Job任务，请使用Python Job"
    # exit()
    rljob = RLJobBuilder(
            # 你运行 ray submit 命令的目录，请不要填写 pcache 路径
            name=args.exp_path.split("/")[-1],
            working_dir=working_dir,
            envs=envs,
            source_root=".",
            main_file="",
            command=command,
            init_command=init_command,
            master=master,
            worker=worker,
            km_conf=KMConf(image=image, retry_strategy=rs),
            data_stores=data_stores,
            tag="type=RLHF,basemodel=MoE_Lite",
            k8s_app_name=args.app,
            k8s_priority=args.priority,  # 配置 GPU 资源优先级
            rdma=True,
            host_network=True,  # 开启 host network, rdma=True 时需要为True。
            enable_pcache_fuse=True,
    )

    rljob.run(enable_wait=False, dima=args.dima)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--app', type=str, default='gbank', help='提交的gpu集群')
    parser.add_argument('--num_node', type=int, default=1, help='节点数')
    parser.add_argument('--gpu_num', type=int, default=1, help='每个节点gpu数')
    parser.add_argument('--gpu_type', type=str, default="a100", help='gpu类型')
    parser.add_argument('--config_path', type=str, default=None, help='配置文件路径')
    parser.add_argument('--exp_path', type=str, default=None, help='实验路径，会用于存放代码，不需要设置')
    parser.add_argument('--nas', type=str, help='nas地址')
    parser.add_argument('--dima', type=str, help='dima id')
    parser.add_argument('--priority', type=str, default='high', help='优先级')
    parser.add_argument('--upload_code', action='store_true', help='是否需要上传代码')
    args = parser.parse_args()
    assert args.exp_path is None, "exp_path会自动注入，不需要手动设置"
    args.exp_path = get_exp_path(args.config_path)


    assert args.app not in ['sgtest'], "sgtest集群不支持Ray Job的提交"
    main(args)


# 使用样例
"""
python run_a100_submit_edl.py \
--app gpu_dev \
--num_node 1 \
--gpu_num 1 \
--config_path OLMo/configs/test/train_1b_0403_2t_ct_wsd_20.yaml \
--exp_path /path/to/llm_pretrain/exp/151724/train_1b_0403_2t_ct_wsd_20_test \
--nas alipay-heyuan-xx-xxxxxx.cn-heyuan-alipay.nas.aliyuncs.com \
--priority high

python run_submit_grpo.py \
    --app llmdebugmalaih800  \
    --num_node 1 \
    --gpu_num 8 \
    --gpu_type h800 \
    --config_path /adabench_nas/sunxiaqing/agent_llm/post_training/rm/scripts/openrlhf/grpo_test.sh \
    --priority high \
    --dima 2025041400108229412 \
    --upload_code 

python run_submit_grpo.py \
    --app lingmlctemp1  \
    --num_node 8 \
    --gpu_num 8 \
    --gpu_type h200 \
    --config_path /adabench_nas/sunxiaqing/agent_llm/post_training/rm/scripts/openrlhf/rl/grpo_generative_rm_thinkrl_mlc.sh \
    --priority high \
    --dima 2025041400108229412 \
    --upload_code

python run_submit_grpo.py \
    --app lings3temp  \
    --num_node 8 \
    --gpu_num 8 \
    --gpu_type h200 \
    --config_path /adabench_nas/sunxiaqing/agent_llm/post_training/rm/scripts/openrlhf/rl/grpo_generative_rm_sft_s3.sh \
    --priority high \
    --dima 2025041400108229412 \
    --upload_code
"""
# end