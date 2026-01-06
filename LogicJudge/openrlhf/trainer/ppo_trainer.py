import os
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional
import json
import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models.utils import actor_group_empty_cache
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils import blending_datasets, get_tokenizer, get_ckpt_path
from openrlhf.utils.tb_utils import keywords_tensorboards
from openrlhf.datasets import PromptDataset, SFTDataset
from datasets import concatenate_datasets
import math
from openrlhf.utils.utils import distributed_order, recover_original_order, sleep_vllm, wakeup_vllm

from .ppo_utils.experience_maker import SingleControllerRemoteExperienceMaker
from .ppo_utils import Experience
import ray
import hashlib
logger = init_logger(__name__)

@ray.remote
def train_model(model_group):
    if model_group is None:
        return {}
    refs = []
    for model in model_group:
        refs.append(model.fit.remote())
    status = ray.get(refs)
    mean_status = {}
    for key in status[0].keys():
        val_list = [statu[key] for statu in status]
        mean_status[key] = sum(val_list) / len(val_list)
    # merge status, use status[0]
    return mean_status



class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(
        self,
        strategy,
        actor_model_group,
        critic_model_group,
        reward_model_groups,
        reference_model_group,
        vllm_engines,
        reward_vllm_engines,
    ) -> None:
        # assert (
        #     not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        # ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_groups = reward_model_groups
        self.reference_model_group = reference_model_group
        self.vllm_engines = vllm_engines
        self.reward_vllm_engines = reward_vllm_engines
        self.tokenizer = get_tokenizer(self.args.pretrain, None)
        self.world_size = len(actor_model_group)

        self.generate_kwargs = {
            "temperature": self.args.temperature,
            'top_p': self.args.top_p,
            'top_k': -1,
            'max_new_tokens': self.args.generate_max_len
        }

        self.experience_maker = SingleControllerRemoteExperienceMaker(
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_groups,
            reference_model_group,
            vllm_engines,
            reward_vllm_engines,
        )

        self.init_visualization()
        
        self.eval_sample_n = self.args.eval_n_samples
        self.prepare_datasets()

    def get_max_steps(self):
        return self.max_steps


    def init_visualization(self):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        strategy = self.strategy
        if self.strategy.args.use_wandb:
            import wandb
            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            self._tensorboard = SummaryWriter(log_dir=self.strategy.args.use_tensorboard)
            print(f"Using tensorboard for logging in {self.strategy.args.use_tensorboard}.")
        else:
            print(f"Don't use tensorboard: {self.strategy.args.use_tensorboard=}, {self._wandb=}")

    
    # 在ExperienceMaker类中包含了具体的实现，在Trainer类中负责数据流的控制，主要为负载均衡
    # Step 0
    def generate_rollout(self, prompts, labels, n_samples_per_prompt):
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in labels], [])

        all_prompts = distributed_order(self.world_size, all_prompts)
        all_labels = distributed_order(self.world_size, all_labels)
        experience_list = self.experience_maker.generate_rollout(all_prompts, all_labels, self.generate_kwargs)
        return experience_list
    
    # Step 1:
    def cal_reward(self, experience_list):
        reward_refs = self.experience_maker.cal_reward(experience_list)
        return reward_refs
    
    # Step 2:
    def cal_base_action_log_probs(self, experience_list):
        base_action_log_probs_ref = self.experience_maker.cal_base_action_log_probs(experience_list)
        return base_action_log_probs_ref

    # Step 3:
    def cal_action_log_probs(self, experience_list):
        action_log_probs_ref = self.experience_maker.cal_action_log_probs(experience_list)
        return action_log_probs_ref

    # Step 4:
    def cal_critic_value(self, experience_list):
        critic_value_ref = self.experience_maker.cal_cirtic_value(experience_list)
        return critic_value_ref

    # Step 5:
    def make_experience(self, experience_list: List[Experience], rewards, base_action_log_probs_list, action_log_probs_list, critic_value):
        rewards = rewards.unsqueeze(-1)
        if critic_value is not None:
            critic_value =  critic_value.unsqueeze(-1)

        for index in range(len(experience_list)):
            experience = experience_list[index]
            experience.info['reward'] = rewards[index]
            
            if action_log_probs_list is not None:
                experience.action_log_probs = action_log_probs_list[index]
                assert experience.action_log_probs.size()[-1] == experience.num_actions, f"{experience.action_log_probs.size()=}***{experience.num_actions=}"

            if base_action_log_probs_list is not None:
                experience.base_action_log_probs = base_action_log_probs_list[index]
                assert experience.base_action_log_probs.size()[-1] == experience.num_actions, f"{experience.base_action_log_probs.size()=}***{experience.num_actions=}"

            if critic_value is not None:
                experience.values = critic_value[index]


        experience_list = recover_original_order(self.world_size, experience_list)
        
        # 确保初始顺序被正确的recover
        if self.strategy.args.n_samples_per_prompt > 1:
            assert experience_list[0].labels == experience_list[self.strategy.args.n_samples_per_prompt-1].labels

        experience_list = self.experience_maker.make_experience(experience_list, self.generate_kwargs)
        experience_list = distributed_order(self.world_size, experience_list)

        refs = []
        # Send experiences to actor model and critic model
        examples_per_actor = int(len(experience_list) // len(self.actor_model_group))
        for idx, actor_model in enumerate(self.actor_model_group):
            refs.append(actor_model.append.remote(experience_list[idx*examples_per_actor: (idx+1)*examples_per_actor]))
        
        if self.critic_model_group is not None:
            examples_per_critic = int(len(experience_list) // len(self.critic_model_group))
            for idx, critic_model in enumerate(self.critic_model_group):
                refs.append(critic_model.append.remote(experience_list[idx*examples_per_critic: (idx+1)*examples_per_critic]))
        ray.get(refs)
        return experience_list

    def set_start_time(self):
        self.start_time = time.time()

    def log_time(self, string=""):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}: {string} consumed {round(time.time() - self.start_time, 3)} seconds.")
        self.start_time = time.time()


    def fit(
        self,
    ) -> None:
        args = self.args

        # broadcast init checkpoint to vllm
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group[0].get_checkpoint_states.remote())
            logger.info(f"checkpoint_states: {checkpoint_states}")
            self._broadcast_to_vllm()
            self.load_filter_datasets(checkpoint_states['episode'])
        else:
            checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)


        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = float("inf") 
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        if self.args.restart_step == 0:
            print("***Eval At Beginning***")
            self.evaluate(0)


        for episode in range(episode, args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
            )

            for rand_prompts, labels in self.prompts_dataloader:
                self.set_start_time() 
                rollout_samples = self.generate_rollout(rand_prompts, labels, self.strategy.args.n_samples_per_prompt)
                self.log_time("Rollout Generation")

                self.set_start_time() 
                rewards_ref = self.cal_reward(rollout_samples)  #...sxq...rollout_samples:512
                if (self.args.colocate_all_models) and ((self.reward_model_groups is not None) or (self.reward_vllm_engines is not None)):
                    ray.get(rewards_ref)
                    if self.reward_model_groups is not None:
                        actor_group_empty_cache(self.reward_model_groups[0])
                    self.log_time("Cal rewards") 
                    self.set_start_time() 

                base_action_log_probs_ref = self.cal_base_action_log_probs(rollout_samples)
                if self.args.colocate_all_models:
                    ray.get(base_action_log_probs_ref)
                    actor_group_empty_cache(self.reference_model_group)
                    self.log_time("Cal reference model probability") 
                    self.set_start_time() 

                action_log_probs_ref = self.cal_action_log_probs(rollout_samples)
                if self.args.colocate_all_models:
                    ray.get(action_log_probs_ref)
                    actor_group_empty_cache(self.actor_model_group)
                    self.log_time("Cal actor model old probability") 
                    self.set_start_time() 
                
                critic_value_ref = self.cal_critic_value(rollout_samples)
                rewards = ray.get(rewards_ref)
                base_action_log_probs_list = ray.get(base_action_log_probs_ref)
                action_log_probs_list = ray.get(action_log_probs_ref)
                critic_value_list = ray.get(critic_value_ref)

                self.log_time("Cal critic value") 
                self.set_start_time() 
                experience_list = self.make_experience(rollout_samples, rewards, base_action_log_probs_list, action_log_probs_list, critic_value_list)
                self.log_time("Make experience") 

                self.set_start_time() 
                train_actor_refs = train_model.remote(self.actor_model_group)
                
                ## 在等待Actor训练的过程中可以做一些写日志的操作，以及根据目前的训练进程做一些动态修改参数的操作
                self.dynamic_max_new_tokens(steps, experience_list)
                self.save_rollout_and_write_mdata_logs(steps, experience_list)

                if args.ref_update_interval != -1 and (steps + 1) % args.ref_update_interval == 0:
                    print(f"[RefModel] Step {steps}: Updating reference model...")
                    actor_state = ray.get(self.actor_model_group[0].get_state_dict.remote())

                    ray.get([
                        ref.set_state_dict.remote(actor_state)
                        for ref in self.reference_model_group
                    ])
                    actor_group_empty_cache(self.reference_model_group)

                if self.args.colocate_all_models:
                    ray.get(train_actor_refs)
                    actor_group_empty_cache(self.actor_model_group)
                    self.log_time("Training actor model") 
                    self.set_start_time() 

                train_critic_refs = train_model.remote(self.critic_model_group)

                actor_status, critic_status = ray.get(train_actor_refs), ray.get(train_critic_refs)
                self.log_time("Training critic model") 
                status = actor_status | critic_status

                # broadcast to vllm
                self.set_start_time() 
                self._broadcast_to_vllm()
                self.log_time("Broadcast vllm") 

                pbar.set_postfix(status)
                # logs/checkpoints
                client_states = {
                    "global_step": steps,
                    "episode": episode,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.set_start_time()
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states, episode)
                self.log_time("Save Logs and Checkpoints") 

                pbar.update()
                steps = steps + 1

            self.save_filter_datasets(episode)
            self.load_filter_datasets(episode + 1)
        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()
    
    def dynamic_max_new_tokens(self, global_step, experience_list):
        tensorboard_step = self.args.restart_step + global_step

        if getattr(self, 'dynamic_mnt_window', None) is None:
            self.dynamic_mnt_window = []

        assert self.args.dynamic_max_new_tokens_percents <= 1, "dynamic_max_new_tokens_percents should be in (0, 1]"
        response_length = []
        for experience in experience_list:
            response_length.append(experience.info['response_length'])
        
        assert len(response_length) == self.args.rollout_batch_size * self.args.n_samples_per_prompt

        response_length_sorted = sorted(response_length)
        max_new_tokens = response_length_sorted[int(len(response_length_sorted) * self.args.dynamic_max_new_tokens_percents)]
        self.dynamic_mnt_window = (self.dynamic_mnt_window + [max_new_tokens])[-10: ]
        max_new_tokens = min(self.args.generate_max_len, max(self.dynamic_mnt_window) + 1000)
        if self.args.dynamic_max_new_tokens_percents > 0:
            self.generate_kwargs['max_new_tokens'] = int(max_new_tokens)
            print(f"Longest Response:{max(response_length_sorted)}; Dynamic max new tokens: {max_new_tokens}")

        if self._tensorboard is not None:
            self._tensorboard.add_scalar(f"generation/longest response length", max(response_length_sorted), tensorboard_step)
            self._tensorboard.add_scalar(f"generation/max new tokens", self.generate_kwargs['max_new_tokens'], tensorboard_step)

    def save_rollout_and_write_mdata_logs(self, global_step, experience_list: List[Experience]):
        experience_list = recover_original_order(self.world_size, experience_list)
        to_write_json = []
        responses = self.experience_maker.tokenizer.batch_decode([experience.sequences for experience in experience_list], skip_special_tokens=False)
        for idx, experience in enumerate(experience_list):
            to_write_json.append(
                {
                    "prompt": experience.prompts,
                    'label': experience.labels,
                    'response': responses[idx].replace(self.experience_maker.tokenizer.pad_token, "").replace(experience.prompts, ""),
                    'reward': experience.info['reward'].item(),
                    'response_length': experience.info['response_length']
                }
            )
        os.makedirs(f"{self.args.save_path}/rollout_output", exist_ok=True) 
        with open(f"{self.args.save_path}/rollout_output/Step{global_step}.json", 'w') as wf:
            wf.write(json.dumps(to_write_json, ensure_ascii=False, indent=2))
            wf.flush()
        
        # no pass ratio for rgrm training
        assert len(to_write_json) == self.strategy.args.rollout_batch_size * self.strategy.args.n_samples_per_prompt
        num_no_pass, correct = 0, 0
        for i in range(self.strategy.args.rollout_batch_size):
            cur_batch = to_write_json[i*self.strategy.args.n_samples_per_prompt:(i+1)*self.strategy.args.n_samples_per_prompt]
            rewards = [item['reward'] for item in cur_batch]
            if rewards.count(-0.5) == self.strategy.args.n_samples_per_prompt:
                num_no_pass += 1
            else:
                correct += rewards.count(1)
        no_path_ratio = num_no_pass / self.strategy.args.rollout_batch_size
        accuracy = correct / (self.strategy.args.rollout_batch_size * self.strategy.args.n_samples_per_prompt)

        if self._tensorboard is not None:
            self._tensorboard.add_scalar("train/no_pass_rate", no_path_ratio, global_step+self.args.restart_step)
            self._tensorboard.add_scalar("train/accuracy", accuracy, global_step+self.args.restart_step)

        if global_step%10 == 0:
            self.write_mdata_logs(to_write_json, global_step)
        
        if self.strategy.args.episode_dynamic_sampling != 0:
            cnt = 0
            for prompt_index in range(self.strategy.args.rollout_batch_size):
                score = 0
                for experience_index in range(self.strategy.args.n_samples_per_prompt):
                    score += experience_list[prompt_index*self.strategy.args.n_samples_per_prompt + experience_index].info['reward'].item()
                score = score / self.strategy.args.n_samples_per_prompt
                if score <= self.strategy.args.episode_dynamic_sampling:
                    cnt += 1
                    self.prompts_buffer.append(experience_list[prompt_index*self.strategy.args.n_samples_per_prompt].prompts)
                    self.labels_buffer.append(experience_list[prompt_index*self.strategy.args.n_samples_per_prompt].labels)

            if self._tensorboard is not None:
                self._tensorboard.add_scalar(f"train/dynamic_filter", self.strategy.args.rollout_batch_size - cnt, global_step+self.args.restart_step)

        keywords_tensorboards(self._tensorboard, responses, global_step+self.args.restart_step)




    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}, episode=0):
        tensorboard_step = self.args.restart_step + global_step
        if global_step % args.logging_steps == 0:
            # TensorBoard
            if self._tensorboard is not None:
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, tensorboard_step)

        if global_step % args.eval_steps == 0:
            self.evaluate(global_step)

        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            ref = [self.actor_model_group[i].save_checkpoint.remote(tag=tag, client_states=client_states) for i in range(len(self.actor_model_group))]
            if self.critic_model_group is not None:
                ref.extend([self.critic_model_group[i].save_checkpoint.remote(tag=tag) for i in range(len(self.critic_model_group))])
            self.save_filter_datasets(episode)
            ray.get(ref)

    def _broadcast_to_vllm(self):
        ray.get([self.actor_model_group[idx].broadcast_to_vllm.remote() for idx in range(len(self.actor_model_group))])
        

    def evaluate(self, global_step):
        # 当且仅当如下条件满足可以evaluate
        # 1. remote reward model; 
        # 2. vllm engine
        tensortboard_step = self.strategy.args.restart_step + global_step
        if self.eval_dataloaders is None:
            return
        self.generate_kwargs['temperautre'] = 0.6
        train_max_new_tokens = self.generate_kwargs['max_new_tokens']
        print(f"""*****Step:{global_step} - Evaluate START*****""")
        eval_data_list = self.args.eval_data.split(",")

        eval_dataloader = self.eval_dataloaders[0]
        
        for max_new_tokens in [4096]:
            self.generate_kwargs['max_new_tokens'] = max_new_tokens
            experience_list = []
            pad_cnt = 0
            cnt_per_reward = 1
            for rand_prompts, labels in eval_dataloader:
                rand_prompts = sum([[prompt for _ in range(self.eval_sample_n)] for prompt in rand_prompts], [])
                labels = sum([[label for _ in range(self.eval_sample_n)] for label in labels], [])
                #  先简化处理了，数据量必须能够整除(len(actor_model_group) * self.args.micro_reward_rollout_batch_size)
                # if self.reward_model_groups is not None:
                cnt_per_reward = len(self.actor_model_group) * self.args.micro_reward_rollout_batch_size
                if len(rand_prompts) % cnt_per_reward != 0:
                    pad_cnt = cnt_per_reward - len(rand_prompts) % cnt_per_reward
                    rand_prompts = rand_prompts + [rand_prompts[0] for _ in range(pad_cnt)]
                    labels = labels + [labels[0] for _ in range(pad_cnt)] 
                print(f"{len(rand_prompts)=}--{len(labels)=}--{pad_cnt=}")
                experience_list = self.generate_rollout(rand_prompts, labels, 1)
                rewards = ray.get(self.cal_reward(experience_list)).unsqueeze(-1)
                for experience, reward in zip(experience_list, rewards):
                    experience.info['reward'] = reward

            experience_list = recover_original_order(self.world_size, experience_list)
            experience_list = experience_list[:len(experience_list)-pad_cnt]

            # for experience in experience_list:
            responses = self.experience_maker.tokenizer.batch_decode([experience.sequences for experience in experience_list], skip_special_tokens=False)
            rewards = [experience.info['reward'] for experience in experience_list]
            response_lengthes = [experience.info['response_length'] for experience in experience_list]
            for dataset_index in range(len(eval_data_list)):
                eval_data_name = eval_data_list[dataset_index].split('/')[-1].split('.')[0]
                start_index = sum(self.eval_datasets_lengthes[:dataset_index])
                end_index = sum(self.eval_datasets_lengthes[:dataset_index + 1])
                print(f"Dataset: {eval_data_name}, {start_index=}, {end_index=}")

                to_write_lines = []
                for index in range(start_index, end_index):
                    to_write_lines.append(
                        {
                            "prompt": experience_list[index].prompts,
                            'label': experience_list[index].labels,
                            'response': responses[index].replace(self.experience_maker.tokenizer.pad_token, "").replace(experience_list[index].prompts, ""),
                            'reward': experience_list[index].info['reward'].item(),
                            'response_length': experience_list[index].info['response_length']
                        }
                    )
                os.makedirs(f"{self.args.save_path}/evaluation_results/Step{global_step}/Max_new_tokens_{max_new_tokens}", exist_ok=True) 
                with open(f"{self.args.save_path}/evaluation_results/Step{global_step}/Max_new_tokens_{max_new_tokens}/Dataset_{eval_data_name}.json", 'w') as wf:
                    wf.write(json.dumps(to_write_lines, indent=2, ensure_ascii=False))
                self.write_mdata_logs(to_write_lines, global_step, eval_data_name)

                dataset_score = sum(rewards[start_index: end_index]) / (end_index - start_index)
                dataset_response_length = sum(response_lengthes[start_index: end_index]) / (end_index - start_index)
                dataset_correct = rewards[start_index: end_index].count(1) / (end_index - start_index)
                dataset_format_fail = rewards[start_index: end_index].count(-1) / (end_index - start_index)

                if self._tensorboard:
                    self._tensorboard.add_scalar(f"eval/reward/{max_new_tokens}/{eval_data_name}", dataset_score, tensortboard_step)
                    self._tensorboard.add_scalar(f"eval/correct/{max_new_tokens}/{eval_data_name}", dataset_correct, tensortboard_step)
                    self._tensorboard.add_scalar(f"eval/format_failure/{max_new_tokens}/{eval_data_name}", dataset_format_fail, tensortboard_step)
                    self._tensorboard.add_scalar(f"eval/response_length/{max_new_tokens}/{eval_data_name}", dataset_response_length, tensortboard_step)
                    self._tensorboard.flush()
                print(f"""\nDataset: {eval_data_name} - Max new tokens: {max_new_tokens}\n""")
                print(f"""AVG Reward:{dataset_score}\nAVG Res Length:{dataset_response_length}\n""")

        self.generate_kwargs['max_new_tokens'] = train_max_new_tokens
        self.generate_kwargs['temperature'] = self.args.temperature
        print("*****Evaluate END*****")

    def save_filter_datasets(self, episode):
        """
        尚未考虑情况：
        如果恰好在两个episode之间还未保存ckpt时任务挂了，会导致少量数据的重写
        """
        args = self.strategy.args
        if args.episode_dynamic_sampling == 0:
            return
        os.makedirs(self.args.save_path, exist_ok=True)

        filter_dataset_path = f"{args.save_path}/episode-{episode}-filter_dataset.jsonl"
        with open(filter_dataset_path, "a+") as wf:
            for prompt, label in zip(self.prompts_buffer, self.labels_buffer):
                wf.write(json.dumps({"prompt": prompt, "label": label}, ensure_ascii=False) + "\n")
        self.prompts_buffer = []
        self.labels_buffer = []

    def load_filter_datasets(self, episode):
        
        if self.strategy.args.episode_dynamic_sampling == 0 or episode==0:
            return

        args = self.strategy.args
        filter_dataset_path = f"{args.save_path}/episode-{episode-1}-filter_dataset.jsonl"
        if not os.path.exists(filter_dataset_path):
            print(f"episode dynamic filtering文件找不到：{filter_dataset_path}")
            return

        raw_dataset = self.prompts_dataloader.dataset
        prompts, labels = [], []
        for line in open(filter_dataset_path).readlines():
            example = json.loads(line)
            prompts.append(example['prompt'])
            labels.append(example['label'])
        raw_dataset.prompts = prompts
        raw_dataset.labels = labels
        self.prompts_dataloader = self.strategy.setup_dataloader(
            raw_dataset, args.rollout_batch_size , True, False
        )
        print(f"成功加载过滤数据：{filter_dataset_path}\n数据量：{len(raw_dataset.prompts)}")


    
    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        prompts_dataset = PromptDataset(
            prompts_data, self.tokenizer, strategy, input_template=args.input_template
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset, args.rollout_batch_size , True, False
        )

        self.max_steps = (
            len(prompts_dataset)
            * args.n_samples_per_prompt
            // args.train_batch_size
            * args.num_episodes
            * args.max_epochs
        )

        self.eval_dataloaders = None
        self.eval_datasets_lengthes = []
        if args.eval_data:
            self.eval_dataloaders = []
            flat_eval_data = []
            eval_data_list = args.eval_data.split(",")
            for eval_data in eval_data_list:
                eval_data = blending_datasets(
                    eval_data,
                    "1.0",
                    strategy,
                    args.seed,
                    max_count=args.max_samples,
                    return_eval=False,
                    train_split=args.prompt_split,
                )
                flat_eval_data.append(eval_data)
                self.eval_datasets_lengthes.append(len(eval_data) * self.eval_sample_n)
            eval_data = concatenate_datasets(flat_eval_data)
            eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
            eval_dataloader = strategy.setup_dataloader(
                eval_dataset, len(eval_data), pin_memory=True, shuffle=False
            )
            self.eval_dataloaders.append(eval_dataloader)
        
        self.prompts_buffer = []
        self.labels_buffer = []

    def write_mdata_logs(self, to_write_lines, step, dataset='Train'):
        import os
        from collections import defaultdict
        prompt_index = defaultdict(int)
        try:
            os.makedirs(f"./logs/case_logs", exist_ok=True)
            wf = open(f"./logs/case_logs/Step{step}-{dataset}.jsonl", 'w')
            for data in to_write_lines:
                result_data = {'dataset_name': dataset, 'category': 0}
                prompt = data['prompt']
                prompt_index[prompt] += 1
                prompt_str = prompt + str(prompt_index[prompt])
                mid = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()
                result_data['mid'] = mid
                result_data['prompt'] = prompt
                result_data['response'] = data['response'].replace(prompt, "")
                result_data['golden_answer'] = data['label']
                result_data['score'] = data['reward']
                result_data['step'] = step
                result_data['model_name'] = self.args.save_path.split("/")[-1]
                result_data['task_name'] = "eval_rollout"
                wf.write(json.dumps(result_data) + "\n")
            wf.flush()
            wf.close()
        except Exception as e:
            print(f"Writing mdata error: {e}")