from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from flash_attn.utils.distributed import all_gather
import ray


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    use_kl_estimator_k3: bool = False,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs.float() - log_probs_base.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # Besides non negative, it is also unbiased and have lower variance.
    if use_kl_estimator_k3:
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    if action_mask is not None:
        kl_reward = -kl_coef * kl
        # The following code is equivalent to:
        #
        # last_reward = torch.zeros_like(kl)
        # for i in range(last_reward.size(0)):
        #     for t in reversed(range(last_reward.size(1))):
        #         if action_mask[i][t] > 0.5:
        #             last_reward[i][t] = r[i]
        #             break
        #
        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

        reward = last_reward + kl_reward
    else:
        # TODO: write a more efficient version
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            kl_reward = -kl_coef * kl_seg
            kl_reward[action_len - 1] += r[i]
            reward.append(kl_reward)

    return reward


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack(
            [torch.logsumexp(l, dim=-1) for l in logits]  # loop to reduce peak mem consumption
        )
        log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def token_level_masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None, loss_coefficient=None) -> torch.Tensor:
    assert loss_coefficient is not None
    if mask is None:
        return tensor.mean(axis=dim) * loss_coefficient
    return (tensor * mask).sum(axis=dim) * loss_coefficient 


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


# Reset positions for packed samples
# For example
# Input: attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
# Output: position_ids  = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def unpacking_samples(values: torch.Tensor, packed_seqlens: list[int]):
    values = values.squeeze(0)
    unpacked_values = []
    offset = 0
    for seqlen in packed_seqlens:
        unpacked_values.append(values[offset : offset + seqlen])
        offset += seqlen
    return unpacked_values



def unpad_and_slice_tensor(sequences, attention_mask, ring_attn_group):
    """
    Unpad and slice tensor for distributed training with ring attention.

    This function performs several operations:
    1. Removes padding, unpads sequences from (batch, seqlen) to (1, total_seqs)
    2. Adapts to ring_attn_group, pads sequences to be divisible by ring_attn_group
    3. Slices the sequences for the current ring_attn_rank

    Example:
        >>> # Input sequences shape: (batch=2, seqlen=4)
        >>> sequences = [[1, 2, 3, 0], [4, 5, 0, 0]]  # 0 is padding
        >>> attention_mask = [[1, 1, 1, 0], [1, 1, 0, 0]]
        >>> # After unpad:
        >>> # sequences: [1, 2, 3, 4, 5]  # shape (1, total_seqs=5)
        >>> # If ring_attn_group size is 2, it will pad to length 6
        >>> # Then slice for current rank (e.g., rank 0 gets [1,2,3], rank 1 gets [4,5,0])

    Args:
        sequences: Input sequences tensor of shape (batch, seqlen)
        attention_mask: Attention mask tensor for the sequences
        ring_attn_group: Ring attention group for distributed processing

    Returns:
        tuple: Processed sequences and related tensors for ring attention
    """
    rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
    sequences, indices, cu_seqlens, _, _ = unpad_input(sequences.unsqueeze(-1), attention_mask)
    sequences = sequences.transpose(0, 1)  # (1, total_seqs)
    rolled_sequences = index_first_axis(
        rearrange(rolled_sequences.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(
        0, 1
    )  # (1, total_seqs)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    position_ids = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(
        0, 1
    )  # (1, total_seqs)
    ring_attn_pad_len = 0
    assert ring_attn_group is None
    # if ring_attn_group is not None:
    #     (sequences, position_ids, rolled_sequences), ring_attn_pad_len = get_tensor_in_current_ring_attn_rank(
    #         [sequences, position_ids, rolled_sequences], ring_attn_group, 0
    #     )
    #     cu_seqlens[-1] += ring_attn_pad_len
    #     update_ring_attn_params(cu_seqlens)
    return sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices


def gather_and_pad_tensor(tensor, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen):
    """
    Gather and pad tensor data (such as logits, log_probs, etc.).

    Example:
        >>> # Input tensor from each rank (shape: (1, local_seq_len))
        >>> # Rank 0: [1, 2, 3]
        >>> # Rank 1: [4, 5, 0]  # 0 is padding
        >>> # After all_gather:
        >>> # tensor: [1, 2, 3, 4, 5, 0]  # shape (1, total_seqs=6)
        >>> # After removing padding (ring_attn_pad_len=1):
        >>> # tensor: [1, 2, 3, 4, 5]  # shape (1, total_seqs=5)
        >>> # After pad_input with original indices:
        >>> # tensor: [[1, 2, 3, 0], [4, 5, 0, 0]]  # shape (batch=2, seqlen=4)

    Args:
        tensor: Input tensor, can be logits, log_probs, etc.
        ring_attn_group: Ring attention group
        ring_attn_pad_len: Padding length
        indices: Indices
        batch: Batch size
        seqlen: Sequence length

    Returns:
        Padded tensor
    """
    assert ring_attn_group is None
    # if ring_attn_group is not None:
    #     tensor = all_gather(tensor.transpose(0, 1), ring_attn_group).transpose(0, 1)  # (1, total_seqs)
    #     if ring_attn_pad_len > 0:
    #         tensor = tensor[:, :-ring_attn_pad_len]
    tensor = pad_input(tensor.transpose(0, 1), indices, batch, seqlen).squeeze(-1)  # (batch, seqlen)
    return tensor


def actor_group_empty_cache(actor_group):
    if actor_group is None:
        return 
    refs = [actor.empty_cache.remote() for actor in actor_group]
    ray.get(refs)
    return 