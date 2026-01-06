from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.distributed
import json


from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples, actor_group_empty_cache
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
from openrlhf.utils.utils import print_gpu_memory_details, get_tokenizer, zero_pad_sequences, wakeup_vllm, sleep_vllm


logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor = None
    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    info: Optional[dict] = None
    kl: Optional[torch.Tensor] = None
    prompts: Optional[str] = None
    labels: Optional[str] = None
    responses: Optional[str] = None
    num_actions: List[int] = None
    
    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, field, to(value, device))
        return self

    # @torch.no_grad()
    # def to_device(self, device: torch.device):
    #     self.sequences = to(self.sequences, device)
    #     self.action_log_probs = to(self.action_log_probs, device)
    #     self.base_action_log_probs = to(self.base_action_log_probs, device)
    #     self.returns = to(self.returns, device)
    #     self.advantages = to(self.advantages, device)
    #     self.values = to(self.values, device)
    #     self.attention_mask = to(self.attention_mask, device)
    #     self.action_mask = to(self.action_mask, device)
    #     self.kl = to(self.kl, device)
    #     self.info = {key: to(value, device) for key, value in self.info.items()}
    #     return self

    # def pin_memory(self):
    #     self.sequences = pin_memory(self.sequences)
    #     self.action_log_probs = pin_memory(self.action_log_probs)
    #     self.base_action_log_probs = pin_memory(self.base_action_log_probs)
    #     self.returns = pin_memory(self.returns)
    #     self.advantages = pin_memory(self.advantages)
    #     self.values = pin_memory(self.values)
    #     self.attention_mask = pin_memory(self.attention_mask)
    #     self.action_mask = pin_memory(self.action_mask)
    #     self.kl = pin_memory(self.kl)
    #     self.info = {key: pin_memory(value) for key, value in self.info.items()}
    #     return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    labels: list[str]


@ray.remote(num_cpus=1)
class RemoteRewardModel:
    def __init__(self, args):
        self.args = args
        from openrlhf.utils import load_func_from_path
        self.remote_rm_url = self.args.remote_rm_url
        self.custom_reward_func = ray.remote(load_func_from_path(self.remote_rm_url, 'reward_func'))

    def get_rewards(self, queries, prompts, labels):
        # batch_size = 64
        num_tasks = min(32, self.args.actor_num_nodes * self.args.actor_num_gpus_per_node)
        batch_size = int(len(queries) // num_tasks)
        r_refs = []
        for i in range(num_tasks):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(queries))
            r = self.custom_reward_func.remote(
                queries[start_idx:end_idx],
                prompts[start_idx:end_idx],
                labels[start_idx:end_idx],
            )
            r_refs.append(r)
        results = ray.get(r_refs)
        return torch.concat(results, dim=0)
    
@ray.remote
def reward_model_forward_batches(reward_model_group, experience_list):
    refs = []
    for idx, experiences in enumerate(experience_list):
        refs.append(reward_model_group[int(idx % len(reward_model_group))].forward.remote(
            experiences
        ))
    rewards = ray.get(refs)
    return torch.concat(rewards, dim=0)
    
@ray.remote
def actor_model_forward_batches(actor_model_group, experience_list: List[Experience]):
    refs = []
    for idx, experiences in enumerate(experience_list):
        refs.append(actor_model_group[int(idx % len(actor_model_group))].forward.remote(
            experiences
        ))
    ref_probs = sum(ray.get(refs), [])
    return ref_probs

@ray.remote
def critic_model_forward_batches(critic_model_group, experience_list):
    refs = []
    for idx, experiences in enumerate(experience_list):
        refs.append(critic_model_group[int(idx % len(critic_model_group))].forward.remote(
            experiences
        ))
    critic_values = ray.get(refs)
    return critic_values

@ray.remote
def aggregation_rm_func(reward_refs, experience_list):
    rewards = ray.get(reward_refs)[0]
    return rewards

@ray.remote
def reward_vllm_forward_batches(reward_vllm_engines, queries, prompts, labels, vllm_reward_format_func, vllm_post_process_func, args):
    vllm_reward_prompts = vllm_reward_format_func(queries, prompts, labels, args)
    wakeup_vllm(reward_vllm_engines, True)
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
    )

    examples_per_vllm =  int((len(vllm_reward_prompts) + len(reward_vllm_engines) - 1) // len(reward_vllm_engines))
    refs = [vllm_engine.add_requests_text_single_controller.remote(sampling_params, vllm_reward_prompts[i * examples_per_vllm: (i + 1) * examples_per_vllm]) for i, vllm_engine in enumerate(reward_vllm_engines)]
    vllm_outputs = sum(ray.get(refs), [])
    reward_vllm_responses = [output.outputs[0].text for output in vllm_outputs]
    input_length = [len(list(output.prompt_token_ids)) for output in vllm_outputs]
    response_length = [len(output.outputs[0].token_ids) for output in vllm_outputs]
    print(f"VLLM Reward Input Length: {sum(input_length)/len(input_length)}")
    print(f"VLLM Reward Response Length: {sum(response_length)/len(response_length)}")
    sleep_vllm(reward_vllm_engines, True)
    rewards = vllm_post_process_func(reward_vllm_responses, queries, prompts, labels, args)
    with open(f"{args.save_path}/llm-as-judge-results.jsonl", "a+") as f:
        for index in range(len(reward_vllm_responses[: 128])):
            f.write(json.dumps({
                'judge result': reward_vllm_responses[index],
                'judge input': vllm_reward_prompts[index],
            }, ensure_ascii=False) + "\n")
    return rewards

@ray.remote
class AdavantageEstimator:
    def __init__(self,
        strategy,
        actor_model_group,
        critic_model_group,
        reward_model_groups,
        reference_model_group,
        vllm_engines,
        reward_vllm_engines=None
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_groups = reward_model_groups
        self.reference_model_group = reference_model_group
        self.advantage_estimator = strategy.args.advantage_estimator
        self.remote_rm_url = self.args.remote_rm_url
        self.vllm_engines = vllm_engines
        self.reward_vllm_engines = reward_vllm_engines
        # hyper_parameters for generation
        self.prompt_max_len = self.args.prompt_max_len
        self.tokenizer = get_tokenizer(self.args.pretrain)


    def make_experience(self, experience_list: List[Experience], generate_kwargs):
        # Set KL
        if (self.reference_model_group is not None) and (not self.strategy.args.use_kl_loss):
            action_log_probs = torch.concat([experience.action_log_probs for experience in experience_list], dim=0)
            base_action_log_probs = torch.concat([experience.base_action_log_probs for experience in experience_list], dim=0)
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=None,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )
            start_index = 0
            for experience in experience_list:
                experience.kl = kl[start_index: start_index + experience.num_actions]
                experience.info['kl'] = torch.mean(experience.kl, dim=-1)
                start_index = experience.num_actions
                
        else:
            for experience in experience_list:
                experience.kl = torch.zeros([experience.num_actions], dtype=torch.float32, device='cpu')
                # experience.kl = torch.zeros_like(experience.sequences, dtype=torch.bfloat16, device='cpu')
                experience.info['kl'] = torch.tensor(0)
        
        # Pre-process Rewards
        # Such as length penalty
        rewards = self.process_rewards(experience_list)
        # Cal Advantages/Returns
        if self.args.advantage_estimator == 'group_norm':
            returns, advantages = self.GRPO_Algo(rewards, experience_list)
        else:
            raise NotImplementedError

        for exp_idx in range(len(experience_list)):
            experience = experience_list[exp_idx]
            experience.advantages = advantages[exp_idx]
            experience.returns = returns[exp_idx]
            experience.info['return'] = experience.returns.sum()
        
        # Post-process advantages
        experience_list = self.process_advantages(experience_list, generate_kwargs)
        
        return experience_list

    def process_advantages(self, experience_list, generate_kwargs):
        self.discard_low_pass_examples(experience_list)
        self.discard_over_long_examples(experience_list, generate_kwargs['max_new_tokens'])
        return experience_list
    

    def discard_low_pass_examples(self, experiences):
        if not self.strategy.args.discard_low_pass_examples:
            return
        discard_low_pass_cnt = 0
        rewards = torch.cat([experience.info["reward"] for experience in experiences])
        rewards = rewards.reshape(-1, self.strategy.args.n_samples_per_prompt)
        group_acc = (rewards.sum(dim=-1, keepdim=True) / self.strategy.args.n_samples_per_prompt).repeat(1, self.strategy.args.n_samples_per_prompt)
        group_acc = group_acc.view(-1)
        rewards = rewards.view(-1)
        for idx, experience in enumerate(experiences):
            if rewards[idx] == 0 and group_acc[idx] <= 0.2-1e-6:
                experience.advantages = torch.zeros_like(experience.advantages)
                experience.returns= torch.zeros_like(experience.returns)
                discard_low_pass_cnt += 1
        print(f"Discard low pass examples: {discard_low_pass_cnt}")

    def discard_over_long_examples(self, experiences, max_new_tokens):
        # 丢弃回复超长的样本。在Reasoning训练中，部分问题由于比较困难，受到Token预算的限制，可能会有能做对的样本由于尚未产生答案给了负分
        if not self.strategy.args.discard_over_long_examples:
            return
        discard_cnt = 0
        for experience in experiences:
            if experience.info['response_length'] >= max_new_tokens:
                experience.advantages = torch.zeros_like(experience.advantages)
                experience.returns = torch.zeros_like(experience.returns)
                discard_cnt += 1
        print(f"Discard over long examples: {discard_cnt}")
    
    def process_rewards(self, experience_list: List[Experience]):
        rewards = torch.cat([experience.info['reward'] for experience in experience_list]).view(-1)

        # Length Penalty
        if self.strategy.args.length_penalty is not None:
            rewards = self.length_penalty(rewards, experience_list)     
        
        # Thinking Penalty
        if self.strategy.args.thinking_penalty:
            rewards = self.thinking_penalty(rewards, experience_list)

        return rewards

    def thinking_penalty(self, rewards, experiences):
        queries = self.tokenizer.batch_decode([experience.sequences for experience in experiences], skip_special_tokens=False)
        thinking_rewards = []
        for query in queries:
            if '<think>' in query or '</think>' in query:
                thinking_rewards.append(-0.5)
            else:
                thinking_rewards.append(0)
        thinking_rewards = torch.tensor(thinking_rewards).to(rewards.device).view(rewards.size())
        return rewards + thinking_rewards


    def length_penalty(self, rewards, experiences):
        rewards = rewards.reshape(-1, self.strategy.args.n_samples_per_prompt)
        lengthes = torch.tensor([experience.info['response_length'] for experience in experiences]).reshape(-1, self.strategy.args.n_samples_per_prompt)
        # if lengthes.max().item() <= 4096:
        #     # 当且仅当最大值超过4096时开启length_penalty
        #     rewards = rewards.view(-1)
        #     return rewards
        if self.strategy.args.length_penalty == "kimi":
            # Kimi length penalty
            # 对于rewards大于0的，对高长度惩罚，对低长度降低，
            # 对于rewards小于0的，对高长度将惩罚
            min_lengthes, _ = lengthes.min(dim=-1, keepdim=True)
            max_lengthes, _ = lengthes.max(dim=-1, keepdim=True)
            lengthes_rewards = 0.5 - (lengthes - min_lengthes) / (max_lengthes - min_lengthes + 1e-9)
            rewards_bool = (rewards == 1) * rewards
            #对于reward
            lengthes_rewards_correct = lengthes_rewards.clone() * rewards_bool
            lengthes_rewards_false = lengthes_rewards.clone() * (1 - rewards_bool) 
            lengthes_rewards_false[lengthes_rewards_false>0] = 0
            lengthes_rewards = lengthes_rewards_correct + lengthes_rewards_false
        elif self.strategy.args.length_penalty == "kimi_correct":
            # Kimi length penalty
            # 对于rewards大于0的，对高长度惩罚，对低长度降低，
            min_lengthes, _ = lengthes.min(dim=-1, keepdim=True)
            max_lengthes, _ = lengthes.max(dim=-1, keepdim=True)
            lengthes_rewards = 0.5 - (lengthes - min_lengthes) / (max_lengthes - min_lengthes + 1e-9)
            rewards_bool = (rewards == 1) * rewards
            #对于reward
            lengthes_rewards = lengthes_rewards.clone() * rewards_bool
        elif self.strategy.args.length_penalty == "high_acc":
            # 动态调节length penalty，对于low accuracy的不进行长度惩罚鼓励更长的探索，只对high accuracy的进行长度惩罚
            min_lengthes, _ = lengthes.min(dim=-1, keepdim=True)
            max_lengthes, _ = lengthes.max(dim=-1, keepdim=True)
            lengthes_rewards = 0.5 - (lengthes - min_lengthes) / (max_lengthes - min_lengthes + 1e-9) 
            #对于reward
            lengthes_rewards_correct = lengthes_rewards.clone() * rewards
            lengthes_rewards_false = lengthes_rewards.clone() * (1 - rewards) 
            lengthes_rewards_false[lengthes_rewards_false>0] = 0
            lengthes_rewards = lengthes_rewards_correct + lengthes_rewards_false
            
            group_acc = (rewards.sum(dim=-1, keepdim=True) / self.strategy.args.n_samples_per_prompt).repeat(1, self.strategy.args.n_samples_per_prompt)
            group_acc[group_acc>=0.5] = 1
            group_acc[group_acc<=0.5] = 0

            lengthes_rewards = lengthes_rewards * group_acc
        elif self.strategy.args.length_penalty == "dynamic":
            group_acc = (rewards.sum(dim=-1, keepdim=True) / self.strategy.args.n_samples_per_prompt).repeat(1, self.strategy.args.n_samples_per_prompt)

            min_lengthes, _ = lengthes.min(dim=-1, keepdim=True)
            max_lengthes, _ = lengthes.max(dim=-1, keepdim=True)
            penalty_efficient = torch.clamp(group_acc, min=0.5)
            lengthes_rewards = penalty_efficient - (lengthes - min_lengthes) / (max_lengthes - min_lengthes + 1e-9) 
            # 对于reward
            lengthes_rewards_correct = lengthes_rewards.clone() * rewards
            lengthes_rewards_false = lengthes_rewards.clone() * (1 - rewards) 
            lengthes_rewards_false[lengthes_rewards_false>0] = 0
            lengthes_rewards = lengthes_rewards_correct + lengthes_rewards_false
        else:
            raise NotImplementedError

        max_lengthes, _ = lengthes.max(dim=-1, keepdim=True)
        lengthes_rewards_mask = max_lengthes>4096
        lengthes_rewards = lengthes_rewards * lengthes_rewards_mask
        rewards = rewards + lengthes_rewards
        rewards = rewards.view(-1)
        return rewards


    def GRPO_Algo(self, rewards, experience_list):
        rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
        raw_rewards = torch.cat([experience.info['reward'] for experience in experience_list]).reshape(-1, self.args.n_samples_per_prompt)
        
        # 将-100视作mask信号, 进行masked normalize rewards
        mask = raw_rewards != -100  # mask 的维度和 x 一样，值为 True 或 False
        masked_rewards = rewards.clone()  # 拷贝一份原始张量
        masked_rewards[~mask] = 0  # 将 mask 中为 False 的位置（即原始值为 -100 的位置）设为 0
        valid_counts = mask.sum(dim=-1, keepdim=True).clamp(min=2)  # 避免出现除零
        mean = masked_rewards.sum(dim=-1, keepdim=True) / valid_counts  # 按 dim=-1 计算均值，忽略 -100
        var = ((masked_rewards - mean) ** 2 * mask).sum(dim=-1, keepdim=True) / (valid_counts-1)   # 按 dim=-1 计算方差
        std = torch.sqrt(var) + 1e-9  # 标准差，避免除零

        #### reinforce 
        rewards = (masked_rewards - mean) / std  # 执行归一化操作
        rewards[~mask] = 0
        # normalize reward
        # rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
        rewards = rewards.reshape(-1)
        
        # 单个值转换为多个值
        reward_list = compute_reward(
                rewards,
                self.args.init_kl_coef,
                [experience.kl for experience in experience_list],
                action_mask=None,
                num_actions=[experience.num_actions for experience in experience_list],
                reward_clip_range=self.args.reward_clip_range,
            )
        returns = self.get_cumulative_returns(
                    reward_list,
                    None,
                    self.strategy.args.gamma,
                )
        advantages = deepcopy(returns)

        for experience in experience_list:
            if experience.info['reward'].item() == -100:
                experience.info['reward'] = torch.zeros_like(experience.info['reward'])
        return returns, advantages

    def Reinforce_Baseline_Algo(self, reward_list):
        pass

    def RLOO_Algo(self, reward_list):
        pass
    
    def PPO_Algo(self):
        pass


    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns



class SingleControllerRemoteExperienceMaker:
    def __init__(
        self,
        strategy,
        actor_model_group,
        critic_model_group,
        reward_model_groups,
        reference_model_group,
        vllm_engines,
        reward_vllm_engines=None
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_groups = reward_model_groups
        self.reference_model_group = reference_model_group
        self.advantage_estimator = strategy.args.advantage_estimator
        self.remote_rm_url = self.args.remote_rm_url
        self.vllm_engines = vllm_engines
        self.reward_vllm_engines = reward_vllm_engines
        self.tokenizer = get_tokenizer(self.args.pretrain)

        # hyper_parameters for generation
        self.prompt_max_len = self.args.prompt_max_len

        # custom reward func for reinforced finetuning
        self.remote_reward_model = None
        if self.remote_rm_url is not None and self.remote_rm_url.endswith(".py"):
            self.remote_reward_model = RemoteRewardModel.remote(self.args)

        self.aggregation_rm_func = aggregation_rm_func
        if getattr(strategy.args, "aggregation_rm_url", None) is not None:
            from openrlhf.utils import load_func_from_path
            self.aggregation_rm_func = ray.remote(load_func_from_path(strategy.args.aggregation_rm_url, 'aggregation_rm_func'))

        if reward_vllm_engines is not None:
            print(f"Loading custom `vllm_reward_format_func(queries, prompts, labels)` from {self.strategy.args.vllm_rm_url}")
            from openrlhf.utils import load_func_from_path
            self.vllm_reward_format_func = load_func_from_path(self.strategy.args.vllm_rm_url, 'vllm_reward_format_func')
            self.vllm_post_process_func = load_func_from_path(self.strategy.args.vllm_rm_url, 'vllm_reward_post_process_func')
        
        self.advantage_estimators = [AdavantageEstimator.remote(
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_groups,
            reference_model_group,
            vllm_engines,
            reward_vllm_engines
        ) for i in range(min(64, int(self.args.rollout_batch_size)))]




    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


    def generate_rollout(self, all_prompts, all_labels, generate_kwargs):
        wakeup_vllm(self.vllm_engines, self.args.vllm_enable_sleep)
        from vllm import SamplingParams
        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=generate_kwargs.get("temperature", 1.0),
            top_p=generate_kwargs.get("top_p", 1.0),
            top_k=generate_kwargs.get("top_k", -1),
            max_tokens=generate_kwargs.get("max_new_tokens", 1024),
            min_tokens=generate_kwargs.get("min_new_tokens", 1),
            skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
            stop=args.stop_list,
        )
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = int((len(all_prompt_token_ids) + len(self.vllm_engines) - 1) // len(self.vllm_engines))
        for i, llm in enumerate(self.vllm_engines):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests_single_controller.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )
        # 必须要先拿到rollout才能有后续的操作
        ray.get(refs)
        sleep_vllm(self.vllm_engines, self.args.vllm_enable_sleep)

        all_outputs = sum(ray.get(refs), [])

        # some settings
        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        max_response_length = generate_kwargs.get("max_new_tokens", 1024)
        truncate_length = self.prompt_max_len + max_response_length

        # Process outputs into Experience objects
        experience_list = []
        for i in range(len(all_outputs)):
            output = all_outputs[i]
            prompt = all_prompts[i]
            label = all_labels[i]

            # Concatenate prompt and output tokens
            input_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
            sequences = torch.tensor(input_ids)
            # response_length = len(output.outputs[0].token_ids) + int(output.outputs[0].token_ids[-1] != eos_token_id)
            response_length = len(output.outputs[0].token_ids)
            total_length = len(input_ids)
            is_clipped = response_length >= max_response_length

            info = {
                "response_length": response_length,
                "total_length": total_length,
                'is_clipped': is_clipped
            }

            experience = Experience(
                sequences=sequences,
                attention_mask=None,
                action_mask=None,
                num_actions=response_length,
                prompts=prompt,
                labels=label,
                info=info,
            )
            experience_list.append(experience)
        return experience_list

    def decode_queries(self, experience_list):
        output_tokens = [experience.sequences for experience in experience_list]
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=False)

    def split_experience_batch(self, experience_list, batch_size):
        experience_batch_list = []
        for i in range(0, len(experience_list), batch_size):
            experience_batch_list.append(experience_list[i : i + batch_size])
        return experience_batch_list
    
    # 目前仅讨论共卡策略, 分布式策略待解决
    def cal_reward(self, experience_list):
        r_refs = []
        
        queries = self.decode_queries(experience_list)
        labels = [experience.labels for experience in experience_list]
        prompts = [experience.prompts for experience in experience_list]
        #首先考虑有没有 Rule-based Reward
        if self.remote_rm_url is not None:
            r_refs.append(self.remote_reward_model.get_rewards.remote(queries, prompts, labels))

        # reward models
        experience_batch_list = self.split_experience_batch(experience_list, self.args.micro_reward_rollout_batch_size)

        if self.reward_model_groups is not None:
            for reward_model_group in self.reward_model_groups:
                r_refs.append(reward_model_forward_batches.remote(reward_model_group, experience_batch_list))
                if self.args.colocate_all_models:
                    ray.get(r_refs[-1])
                    actor_group_empty_cache(reward_model_group)
        
        if self.reward_vllm_engines is not None:
            if self.args.colocate_all_models and self.reward_model_groups is not None:
                ray.get(r_refs[-1])
                actor_group_empty_cache(self.reward_model_groups[-1])
            r_refs.append(reward_vllm_forward_batches.remote(self.reward_vllm_engines, 
                queries, prompts, labels, 
                self.vllm_reward_format_func, 
                self.vllm_post_process_func, self.args)
            )
        return self.aggregation_rm_func.remote(r_refs, experience_list)
    
    def cal_base_action_log_probs(self, experience_list):
        if self.reference_model_group is None:
            return ray.put(None)

        # reference model
        experience_batch_list = self.split_experience_batch(experience_list, self.args.micro_reference_rollout_batch_size)
        base_action_log_probs_ref = actor_model_forward_batches.remote(self.reference_model_group, experience_batch_list)
        return base_action_log_probs_ref
    
    def cal_action_log_probs(self, experience_list):
        # actor model
        # 在（Onpolicy，KL=0）时，我们不需要计算old probs
        if (self.args.train_batch_size == self.args.n_samples_per_prompt * self.args.rollout_batch_size) and (self.strategy.args.use_kl_loss or self.reference_model_group is None):
            return ray.put(None)

        experience_batch_list = self.split_experience_batch(experience_list, self.args.micro_reference_rollout_batch_size)
        action_log_probs_ref = actor_model_forward_batches.remote(self.actor_model_group, experience_batch_list)
        return action_log_probs_ref

    def cal_cirtic_value(self, experience_list):
        if self.critic_model_group is None:
            return ray.put(None)

        # critic_model
        experience_batch_list = self.split_experience_batch(experience_list, self.args.micro_reference_rollout_batch_size)
        critic_value_ref = critic_model_forward_batches.remote(self.critic_model_group, experience_batch_list)
        return critic_value_ref
    
    def make_experience(self, experience_list: List[Experience], generate_kwargs):
        # Set KL
        examples_per_estimator = int(len(experience_list) // len(self.advantage_estimators))
        refs = []
        for index, advantage_estimator in enumerate(self.advantage_estimators):
            refs.append(advantage_estimator.make_experience.remote(experience_list[index * examples_per_estimator : (index + 1) * examples_per_estimator], generate_kwargs))
        return sum(ray.get(refs), [])

