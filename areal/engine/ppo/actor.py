import functools
import json
from typing import Any
import math
import torch

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.megatron_engine import MegatronEngine
from areal.utils import logging, stats_tracker
from areal.utils.data import (
    KLEstimator,
    Normalization,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import (
    dynamic_sampling,
    gather_logprobs,
    gather_logprobs_entropy,
    ppo_actor_loss_fn,
    reward_overlong_penalty,
)
from areal.utils.perf_tracer import trace_perf

from typing import List, Callable, Optional

logger = logging.getLogger(__name__)



class CosineScaledSparseReward(object):
    """
    Cosine-scaled sparse reward transformation with configurable reshape level.
    
    Supports two reshape levels:
    - "batch": compute statistics across entire batch (default)
    - "group": compute statistics within fixed-size groups
    
    Args:
        reshape_level: "batch" or "group", determines scope of statistics computation
        group_size: Required if reshape_level="group", size of each group
        upper_bound_only: If True, only reshape correct sequences (score == 1)
    """
    
    def __init__(
        self,
        reshape_level: str = "batch",
        group_size: int | None = None,
        upper_bound_only: bool = True,
    ):
        if reshape_level not in {"batch", "group"}:
            raise ValueError(
                f"reshape_level must be 'batch' or 'group', got {reshape_level}"
            )
        if reshape_level == "group" and group_size is None:
            raise ValueError("group_size must be provided if using group-level reshape")
        
        self.reshape_level = reshape_level
        self.group_size = group_size
        self.upper_bound_only = upper_bound_only
    
    def _process_slice(
        self,
        gen_lengths: torch.Tensor,
        reward_scores: torch.Tensor,
        upper_bound_only: bool = True,
    ) -> torch.Tensor:
        """
        Apply cosine-scaled sparse reward transformation to a slice of data.
        
        This is the core logic that can be applied to either the full batch
        or individual groups.
        
        Args:
            gen_lengths: Tensor of shape [slice_size] containing generation lengths
            reward_scores: Tensor of shape [slice_size] containing reward scores
            upper_bound_only: If True, only reshape correct sequences (score == 1)
            
        Returns:
            Tensor of shape [slice_size] with transformed rewards
        """
        device = reward_scores.device
        dtype = reward_scores.dtype
        
        # Find max and min sparse values (scores that are not 0 or 1)
        sparse_mask = (reward_scores != 0) & (reward_scores != 1)
        if sparse_mask.any():
            max_value_sparse = reward_scores[sparse_mask].max()
            min_value_sparse = reward_scores[sparse_mask].min()
        else:
            max_value_sparse = torch.tensor(0.0, device=device, dtype=dtype)
            min_value_sparse = torch.tensor(1.0, device=device, dtype=dtype)
        
        # Find max lengths for correct (score == 1) and wrong (score == 0) sequences
        correct_mask = reward_scores == 1
        wrong_mask = reward_scores == 0
        
        if correct_mask.any():
            max_len_correct = gen_lengths[correct_mask].max().float()
        else:
            max_len_correct = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        if wrong_mask.any():
            max_len_wrong = gen_lengths[wrong_mask].max().float()
        else:
            max_len_wrong = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        # Initialize rewards with original scores
        rewards = reward_scores.clone()
        
        # Process correct sequences (score == 1)
        if correct_mask.any() and max_len_correct.item() > 0:
            correct_gen_lengths = gen_lengths[correct_mask].float()
            progress = correct_gen_lengths / max_len_correct
            cosine = torch.cos(progress * math.pi)
            min_value = max_value_sparse
            max_value = torch.tensor(1.0, device=device, dtype=dtype)
            rewards[correct_mask] = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        
        # Process wrong sequences (score == 0)
        if not upper_bound_only:
            if wrong_mask.any() and max_len_wrong.item() > 0:
                wrong_gen_lengths = gen_lengths[wrong_mask].float()
                progress = wrong_gen_lengths / max_len_wrong
                cosine = torch.cos(progress * math.pi)
                # Yes, they are swapped. This is required for the cosine formula below
                # to work with negative numbers.
                min_value = min_value_sparse
                max_value = torch.tensor(0.0, device=device, dtype=dtype)
                rewards[wrong_mask] = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        
        return rewards
    
    def reward(
        self,
        gen_lengths: torch.Tensor,
        reward_scores: torch.Tensor,
        upper_bound_only: bool | None = None,
    ) -> torch.Tensor:
        """
        Apply cosine-scaled sparse reward transformation.
        
        Args:
            gen_lengths: Tensor of shape [batch_size] containing generation lengths
            reward_scores: Tensor of shape [batch_size] containing reward scores
            upper_bound_only: If provided, overrides the instance default. If True, only reshape correct sequences (score == 1)
            
        Returns:
            Tensor of shape [batch_size] with transformed rewards
        """
        bs = gen_lengths.size(0)
        
        # Use instance default if not provided
        if upper_bound_only is None:
            upper_bound_only = self.upper_bound_only
        
        if self.reshape_level == "batch":
            # Apply transformation to entire batch at once
            return self._process_slice(gen_lengths, reward_scores, upper_bound_only)
        
        elif self.reshape_level == "group":
            # Apply transformation to each group independently
            rewards = torch.zeros_like(reward_scores)
            for i in range(0, bs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                gen_lengths_slice = gen_lengths[s]
                reward_scores_slice = reward_scores[s]
                
                group_rewards = self._process_slice(
                    gen_lengths_slice, reward_scores_slice, upper_bound_only
                )
                rewards[s] = group_rewards
            
            return rewards


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.group_size = config.group_size

        self.kl_ctl = config.kl_ctl
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )
        self.positive_adv_weight = config.positive_adv_weight
            
        # Initialize cosine-scaled sparse reward transformation
        if config.reward_cosine_sparse_config and config.reward_cosine_sparse_config.enabled:
            self.reward_sparser = CosineScaledSparseReward(
                reshape_level=config.reward_cosine_sparse_config.reshape_level,
                group_size=(
                    config.reward_cosine_sparse_config.group_size
                    if config.reward_cosine_sparse_config.reshape_level == "group"
                    else None
                ),
                upper_bound_only=config.reward_cosine_sparse_config.upper_bound_only,
            )
        else:
            self.reward_sparser = None
        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature
        self.dynamic_sampling = config.dynamic_sampling

        self.m2_threshold = config.m2_threshold

        # Log critical GSPO/GRPO configuration for reproducibility
        logger.info("PPOActor Configuration:")
        logger.info(
            f"  importance_sampling_level: {getattr(config, 'importance_sampling_level', 'NOT SET (defaults to token)')}"
        )
        logger.info(
            f"  adv_norm: {config.adv_norm if config.adv_norm else 'DISABLED (None)'}"
        )
        logger.info(
            f"  reward_norm: {config.reward_norm if config.reward_norm else 'DISABLED (None)'}"
        )
        logger.info(f"  eps_clip: {config.eps_clip}")
        logger.info(f"  group_size: {config.group_size}")
        logger.info(f"  cosine reward reshape: {config.reward_cosine_sparse_config}")
        logger.info(
            f"  positive_adv_weight: {config.positive_adv_weight if config.positive_adv_weight is not None else 'DISABLED (None)'}"
        )
        logger.info(
            f"  enable_turns_advantage_scaling: {config.enable_turns_advantage_scaling}"
        )
        if config.enable_turns_advantage_scaling:
            logger.info(
                f"    turns_advantage_penalty: {config.turns_advantage_penalty}"
            )
            logger.info(
                f"    turns_advantage_min_turns: {config.turns_advantage_min_turns}"
            )
        logger.info(
            f"  enable_focal_advantage_scaling: {config.enable_focal_advantage_scaling}"
        )
        if config.enable_focal_advantage_scaling:
            logger.info(
                f"    focal_advantage_gamma: {config.focal_advantage_gamma}"
            )
            logger.info(
                f"    focal_advantage_epsilon: {config.focal_advantage_epsilon}"
            )
    def focal_advantage_scaling(
        self, advantages: torch.Tensor, loss_mask: torch.Tensor, data: dict[str, Any]
    ) -> torch.Tensor:
        """
        Apply focal advantage scaling based on group difficulty score.
        
        This method computes focal weights based on the difficulty of each group (prompt),
        where harder groups (lower mean rewards) receive higher weights.
        Reference: CosineScaledSparseReward for group-level computation pattern.
        
        Args:
            advantages: Tensor of shape [batch_size, seq_len] containing advantages
            loss_mask: Tensor of shape [batch_size, seq_len] containing loss mask
            data: Data dictionary containing rewards and other information
            
        Returns:
            Weighted advantages of shape [batch_size, seq_len]
        """
        # Get original reward scores (each sample has one reward value in [0, 1])
        # Use rewards_original which is saved before transformation (scaling, bias, clip, norm)
        # to ensure values are in [0, 1] range for focal advantage calculation
        reward_score = data.get("rewards_original", data.get("rewards", None))
        if reward_score is None:
            logger.warning(
                "No rewards found in data for focal advantage scaling, returning original advantages"
            )
            return advantages
        
        bs = reward_score.shape[0]
        
        # Compute mean rewards per group (similar to CosineScaledSparseReward)
        # group_difficulty_score represents how well this group performed
        # Higher score means easier (better performance)
        if self.group_size > 1:
            # Group-level computation: compute statistics within each group
            num_groups = bs // self.group_size
            mean_rewards = torch.zeros(bs, 1, device=reward_score.device, dtype=reward_score.dtype)
            
            for i in range(num_groups):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                reward_slice = reward_score[s]  # [group_size]
                
                # Compute mean for this group (group_difficulty_score)
                group_mean = reward_slice.mean()
                
                # Broadcast to all samples in this group
                mean_rewards[s] = group_mean.unsqueeze(0)
        else:
            # When group_size=1, each sample is its own group
            # group_difficulty_score is just the reward_score itself
            mean_rewards = reward_score.unsqueeze(1)  # [batch_size, 1]
        
        # Compute adaptive weights (Focal Weight)
        # Following the pseudo-code: group_difficulty_score = mean_rewards
        # Since rewards are already in [0, 1], no need for sigmoid normalization
        group_difficulty_score = mean_rewards
        
        # Handle boundary cases: skip focal weight calculation for extreme values
        # If reward mean is 0, set advantage to -1
        # If reward mean is 1, set advantage to +0.1
        mean_rewards_expanded = mean_rewards.expand_as(advantages)  # [batch_size, seq_len]
        
        # Create masks for boundary cases
        mean_is_zero = (mean_rewards_expanded == 0.0)
        mean_is_one = (mean_rewards_expanded == 1.0)
        boundary_mask = mean_is_zero | mean_is_one
        
        # Initialize weighted_advantages with original advantages
        weighted_advantages = advantages.clone()
        
        # Apply boundary case values
        weighted_advantages = torch.where(
            mean_is_zero,
            torch.full_like(advantages, -1.0),
            weighted_advantages
        )
        weighted_advantages = torch.where(
            mean_is_one,
            torch.full_like(advantages, 0.1),
            weighted_advantages
        )
        
        # For non-boundary cases, compute and apply focal weights
        if not boundary_mask.all():
            # Focal Weight Calculation
            # Simple samples (score -> 1), weight -> 0
            # Hard samples (score -> 0), weight -> 1
            # Clamp to prevent numerical issues
            gamma = self.config.focal_advantage_gamma
            adaptive_weights = (1 - group_difficulty_score).clamp(min=0.1, max=1.0) ** gamma
            
            # Apply weights to advantages
            # Expand weights from [batch_size, 1] to [batch_size, seq_len]
            adaptive_weights_expanded = adaptive_weights.expand_as(advantages)
            
            # Apply focal weights only to non-boundary cases
            non_boundary_mask = ~boundary_mask
            weighted_advantages = torch.where(
                non_boundary_mask,
                advantages * adaptive_weights_expanded,
                weighted_advantages
            )
        
        # Log statistics
        stats_tracker.get("ppo_actor").scalar(
            focal_weight_mean=adaptive_weights.mean().item(),
            focal_weight_max=adaptive_weights.max().item(),
            focal_weight_min=adaptive_weights.min().item(),
            group_difficulty_mean=group_difficulty_score.mean().item(),
        )
        
        return weighted_advantages

    @trace_perf("ppo_actor.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(
        self,
        data: dict[str, Any],
        temperature: float | None = None,
    ) -> torch.Tensor | None:
        def calc_logprobs(logits, input_data):
            labels = input_data.get(
                "rolled_input_ids",
                torch.roll(input_data["input_ids"], shifts=-1, dims=-1),
            )
            logprobs = gather_logprobs(logits, labels, temperature or 1.0)
            return logprobs

        self.engine.eval()
        return self.engine.forward(
            input_=data,
            post_hook=calc_logprobs,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    @trace_perf("ppo_actor.compute_advantages", category="compute")
    def compute_advantages(self, data: dict[str, Any]) -> None:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        
        reward_score = data["rewards"]

        # Store original rewards before transformation for focal advantage scaling
        # Original rewards should be in [0, 1] range
        data["rewards_original"] = reward_score.clone()

        # Reward Sparse
        if self.reward_sparser is not None:
            # Calculate generation lengths for each sequence
            # gen_lengths is the number of generated tokens (loss_mask marks generated tokens)
            gen_lengths = data["loss_mask"].sum(-1)
            # Apply sparse reward transformation (now accepts tensors directly)
            reward_score = self.reward_sparser.reward(
                gen_lengths=gen_lengths,
                reward_scores=reward_score,
            )

        # Reward Scaling
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)

        # Store transformed rewards in a separate field for dynamic_sampling
        # This preserves the original rewards for logging while allowing dynamic_sampling
        # to operate on transformed rewards (e.g., after reward_cosine_sparse transformation)
        data["rewards_for_sampling"] = reward_score

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            old_logp = data["logprobs"] = data["prox_logp"]
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp", torch.zeros_like(old_logp))
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            # Skip tokens that do not contribute to the loss
            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        data["returns"] = advantages + values

        # # Optionally perform advantage normalization.
        # if self.adv_norm is not None:
        #     advantages = self.adv_norm(advantages, loss_mask)

        # Apply Focal Advantage Scaling
        if self.config.enable_focal_advantage_scaling:
            # Store rewards for focal computation
            data["rewards_for_computation"] = rewards
            advantages = self.focal_advantage_scaling(advantages, loss_mask, data)

        # Apply turns-based advantage scaling (only if enabled in config)
        # Scaling coefficient: 1.0 at min_turns, decreasing by penalty_per_turn for each additional turn
        turns = data.get("turns", None)
        if self.config.enable_turns_advantage_scaling and turns is not None:
            if not isinstance(turns, torch.Tensor):
                turns = torch.tensor(turns, dtype=torch.long, device=advantages.device)
            else:
                turns = turns.to(device=advantages.device)
            
            # Ensure shape matches batch size
            if turns.shape[0] == bs:
                # Calculate scaling using config parameters
                penalty_per_turn = self.config.turns_advantage_penalty
                min_turns = self.config.turns_advantage_min_turns
                turns_scaling = 1.0 - penalty_per_turn * (turns.float() - min_turns)
                # Clamp to reasonable range [0.1, 1.0]
                turns_scaling = torch.clamp(turns_scaling, min=0.1, max=1.0)
                # Log turns_scaling statistics
                stats_tracker.get("ppo_actor").scalar(
                    turns_scaling_mean=turns_scaling.mean().item(),
                    turns_scaling_max=turns_scaling.max().item(),
                    turns_scaling_min=turns_scaling.min().item(),
                )
                # Expand to match advantages shape [batch, seq_len]
                turns_scaling = turns_scaling.unsqueeze(1).expand_as(advantages)
                # Only apply scaling to positive advantages, keep negative ones unchanged
                advantages = torch.where(
                    advantages > 0,
                    advantages * turns_scaling,
                    advantages,
                )
        
        # Apply weight to advantages for positive samples
        # Positive samples are identified by 'is_positive_sample' tensor field in data dict
        # This field should be a 1D tensor of shape [batch_size] with 1 for positive samples, 0 for negative
        if self.positive_adv_weight is not None:
            is_positive_sample = data.get("is_positive_sample", None)
            if is_positive_sample is not None:
                # Ensure it's a tensor and has the correct shape
                if not isinstance(is_positive_sample, torch.Tensor):
                    is_positive_sample = torch.tensor(
                        is_positive_sample, dtype=torch.bool, device=advantages.device
                    )
                else:
                    is_positive_sample = is_positive_sample.to(device=advantages.device)
                
                # Ensure shape matches batch size
                if is_positive_sample.shape[0] != bs:
                    logger.warning(
                        f"is_positive_sample shape {is_positive_sample.shape[0]} doesn't match batch size {bs}, skipping positive sample weighting"
                    )
                else:
                    # Convert to bool mask if needed
                    if is_positive_sample.dtype != torch.bool:
                        positive_mask = is_positive_sample.bool()
                    else:
                        positive_mask = is_positive_sample
                    
                    # Apply weight to advantages for positive samples
                    # Expand mask to match advantage shape [batch, seq_len]
                    positive_mask_expanded = positive_mask.unsqueeze(1).expand_as(advantages)
                    advantages = torch.where(
                        positive_mask_expanded,
                        advantages * self.positive_adv_weight,
                        advantages,
                    )
            else:
                logger.warning(
                    "positive_adv_weight is set but 'is_positive_sample' field not found in data dict, skipping positive sample weighting"
                )
        # Optionally perform advantage normalization.
        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask)
        # Class-Aware Advantage reshape
        

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp

    @trace_perf("ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(self, data: dict[str, Any]) -> None:
        # Record original task_reward before dynamic_sampling
        dynamic_sampling_applied = False
        orig_reward_score = None
        orig_f1_first_turn = None
        with stats_tracker.scope("dynamic_sampling"):
            if self.dynamic_sampling and len(data["rewards"]) % self.group_size == 0:
                # Save original data before dynamic_sampling filters them
                orig_reward_score = data["rewards"]
                if "f1_first_turn" in data:
                    orig_f1_first_turn = data["f1_first_turn"]
                
                data, sampling_stat = dynamic_sampling(data, self.group_size)
                stats_tracker.scalar(**sampling_stat)
                dynamic_sampling_applied = True

        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=torch.ones_like(loss_mask, dtype=torch.bool),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )
        # If dynamic_sampling was applied, record task_reward with original (pre-sampling) data
        # using a separate denominator, and task_reward_ds with post-sampling data
        if dynamic_sampling_applied:
            seq_stats["task_reward_ds"] = reward_score.float()
            # Record original task_reward with its own denominator (different batch size)
            orig_n_seqs = torch.ones_like(orig_reward_score, dtype=torch.bool)
            stats_tracker.denominator(orig_n_seqs=orig_n_seqs)
            stats_tracker.stat(task_reward=orig_reward_score.float(), denominator="orig_n_seqs")
        else:
            seq_stats["task_reward"] = reward_score.float()
        # Add f1_first_turn if available in data
        # If dynamic_sampling was applied, record with original (pre-sampling) data
        if dynamic_sampling_applied and orig_f1_first_turn is not None:
            f1_first_turn = orig_f1_first_turn
            if not isinstance(f1_first_turn, torch.Tensor):
                f1_first_turn = torch.tensor(
                    f1_first_turn, dtype=torch.float32, device=reward_score.device
                )
            # Record with orig_n_seqs denominator (already registered above)
            stats_tracker.stat(f1_first_turn=f1_first_turn.float(), denominator="orig_n_seqs")
        elif "f1_first_turn" in data:
            f1_first_turn = data["f1_first_turn"]
            if not isinstance(f1_first_turn, torch.Tensor):
                f1_first_turn = torch.tensor(
                    f1_first_turn, dtype=torch.float32, device=reward_score.device
                )
            seq_stats["f1_first_turn"] = f1_first_turn.float()
        # Add turns if available in data
        if "turns" in data:
            turns = data["turns"]
            if not isinstance(turns, torch.Tensor):
                turns = torch.tensor(
                    turns, dtype=torch.float32, device=reward_score.device
                )
            seq_stats["turns"] = turns.float()
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.behav_imp_weight_cap is not None:
            scalars["behav_imp_weight_cap"] = self.config.behav_imp_weight_cap
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )
        ########## Logging code ends ##########

        for key in ["rewards", "tot_rewards", "kl_rewards", "versions"]:
            data.pop(key, None)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_loss_fn,
                        temperature=self.temperature,
                        eps_clip=self.config.eps_clip,
                        eps_clip_higher=self.config.eps_clip_higher,
                        c_clip=self.config.c_clip,
                        behav_imp_weight_cap=self.config.behav_imp_weight_cap,
                        m2_threshold=self.m2_threshold,
                        importance_sampling_level=self.config.importance_sampling_level,
                        seq_is_ratio_max=self.config.seq_is_ratio_max,
                        seq_is_ratio_min=self.config.seq_is_ratio_min,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)


class FSDPPPOActor(FSDPEngine):
    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> None:
        self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)


class MegatronPPOActor(MegatronEngine):
    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> None:
        self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)


def grpo_loss_fn(
    logits: torch.Tensor,
    input_data: dict,
    temperature: float,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    seq_is_ratio_max: float | None = None,
    seq_is_ratio_min: float | None = None,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    # Use rolled input_ids. Ulysses SP will roll input_ids in ulysses_prepare_inputs().
    labels = input_data.get(
        "rolled_input_ids",
        torch.roll(input_data["input_ids"], shifts=-1, dims=-1),
    )
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    # Use full loss_mask. Ulysses SP will slice loss_mask in ulysses_prepare_inputs().
    loss_mask = input_data.get("full_loss_mask", input_data["loss_mask"]).bool()
    prox_logp = input_data["prox_logp"]

    logprobs, entropy = gather_logprobs_entropy(logits, labels, temperature)
    entropy = entropy.detach()

    # If m2_threshold is set, use M2PO loss function.
    if m2_threshold is not None:
        delta = old_logp - prox_logp
        m2 = delta * delta
        mask_flat = loss_mask.view(-1)
        m2_selected = m2.view(-1)[mask_flat]
        if m2_selected.numel() == 0:
            full_loss_mask = loss_mask
        else:
            sorted_m2, indices = torch.sort(m2_selected, descending=True)
            restored_indices = torch.argsort(indices)
            sorted_m2_loss_mask = get_m2po_loss_mask(
                sorted_m2=sorted_m2, m2_threshold=m2_threshold
            )
            m2_selected_mask = sorted_m2_loss_mask[restored_indices]
            m2_full_flat = torch.zeros_like(
                mask_flat, dtype=torch.bool, device=loss_mask.device
            )
            m2_full_flat[mask_flat] = m2_selected_mask
            full_loss_mask = m2_full_flat.view_as(loss_mask)
        loss_mask = full_loss_mask

    loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
        c_clip=c_clip,
        proximal_logprobs=prox_logp,
        behav_imp_weight_cap=behav_imp_weight_cap,
        importance_sampling_level=importance_sampling_level,
        cu_seqlens=input_data.get("cu_seqlens"),
        seq_is_ratio_max=seq_is_ratio_max,
        seq_is_ratio_min=seq_is_ratio_min,
    )

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )
    # Log sequence-level IS ratio filtering stats if available
    if "seq_is_filter" in stat:
        filter_stats = stat["seq_is_filter"]
        stats_tracker.scalar(
            seq_is_n_total=filter_stats["n_seq_total"],
            seq_is_n_kept=filter_stats["n_seq_kept"],
            seq_is_n_filtered_high=filter_stats["n_seq_filtered_high"],
            seq_is_n_filtered_low=filter_stats["n_seq_filtered_low"],
            seq_is_ratio_max=filter_stats["seq_is_ratio_max"],
            seq_is_ratio_min=filter_stats["seq_is_ratio_min"],
        )
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    clip_mask = stat["clip_mask"]
    clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
    clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
    stats_tracker.stat(
        clipped_new_logp=clipped_new_logp,
        clipped_old_logp=clipped_old_logp,
        denominator="clipped_tokens",
    )
    return loss


def get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.
    Mask the tokens whose second-momentum is the largest, until the average second-momentum is below the threshold.
    """
    n = sorted_m2.numel()
    if n == 0:
        return torch.ones_like(sorted_m2, dtype=torch.bool)

    # Suffix sums: S[i] = sum(sorted_m2[i:])
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)

    # Number of elements in suffix: N[i] = n - i
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)

    # Average of suffix: A[i] = S[i] / N[i]
    avg_m2_suffix = suffix_sums / counts

    # Find the first index `k` where the average of the rest is below threshold.
    below_threshold_indices = torch.where(avg_m2_suffix < m2_threshold)[0]

    if len(below_threshold_indices) > 0:
        num_to_mask = below_threshold_indices[0].item()
    else:
        # All suffix averages are >= threshold. Mask all but one to satisfy assertion.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    if loss_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when getting the m2po loss mask.")

    return loss_mask
