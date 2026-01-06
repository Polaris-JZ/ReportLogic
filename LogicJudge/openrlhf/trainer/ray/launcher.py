import logging
import os
import socket
from typing import Callable, Dict, List, Optional, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.trainer.ray.utils import ray_noset_visible_devices
from openrlhf.utils.deepspeed import DeepspeedStrategy
from aistudio_checkpoint.aistudio_base_checkpointer import lazy_load_checkpoint


class DistributedTorchRayActor:
    def __init__(self, world_size, rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
        # environment variable for each actor, unless
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set, so
        # set local rank to 0 when the flag is not applicable.
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BasePPORole(DistributedTorchRayActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()

    def empty_cache(self):
        torch.cuda.empty_cache()


@ray.remote(num_gpus=1)
class ReferenceModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        if isinstance(pretrain, str) and pretrain.startswith("aistudio"):
            pretrain = lazy_load_checkpoint(resuming_path=pretrain)
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            packing_samples=strategy.args.packing_samples,
        )
        model.unembed_ckpt = strategy.args.unembed_ckpt
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
            self,   
            experiences
    ) -> torch.Tensor:
        device = torch.cuda.current_device()

        # packing
        position_ids = torch.cat([torch.arange(experience.sequences.size()[-1]) for experience in experiences], dim=0).unsqueeze(0)
        packed_seq_lens = [experience.sequences.size()[-1] for experience in experiences]
        sequences = torch.cat([experience.sequences for experience in experiences], dim=0).unsqueeze(0)
        num_actions = [experience.num_actions for experience in experiences]

        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                num_actions = num_actions,
                attention_mask = None,
                return_output=False,
                packed_seq_lens=packed_seq_lens,
                position_ids = position_ids.to(device)
            )
        
        log_probs = log_probs.to("cpu")
        log_probs_list = []
        start_idx = 0
        assert sum(num_actions) == log_probs.size()[-1]
        for num_action in num_actions:
            end_idx = start_idx + num_action
            log_probs_list.append(log_probs[:, start_idx: end_idx].squeeze(0))
            start_idx = end_idx
        return log_probs_list

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
    
    def set_chunk_size(self, chunk_size):
        self.model.chunk_size = chunk_size

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True)


@ray.remote(num_gpus=1)
class RewardModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        if isinstance(pretrain, str) and pretrain.startswith("aistudio"):
            pretrain = lazy_load_checkpoint(resuming_path=pretrain)
                
        model = get_llm_for_sequence_regression(
            pretrain,
            "reward",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            value_head_prefix=strategy.args.value_head_prefix,
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self, experiences
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        assert self.strategy.args.packing_samples, "Not-packing is implemented in SingleController"

        # packing
        position_ids = torch.cat([torch.arange(experience.sequences.size()[-1]) for experience in experiences], dim=0).unsqueeze(0)
        packed_seq_lens = [experience.sequences.size()[-1] for experience in experiences]
        sequences = torch.cat([experience.sequences for experience in experiences], dim=0).unsqueeze(0)

        with torch.no_grad():
            reward = self.model(
                sequences.to(device), 
                position_ids=position_ids.to(device), 
                attention_mask=None, 
                packed_seq_lens=packed_seq_lens
                )
            
        return reward.to("cpu")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


class PPORayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BasePPORole]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BasePPORole],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(self._num_nodes * self._num_gpus_per_node)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank,
                        ),
                    ).remote(world_size, rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_save_model(self):
        """Save actor model on rank 0.

        Returns:
            List: list of remote object refs.
        """
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs

    def __len__(self):
        return len(self._actor_handlers)
    
    def __getitem__(self, index):
        return self._actor_handlers[index]