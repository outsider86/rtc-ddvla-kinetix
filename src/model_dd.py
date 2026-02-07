"""Discrete diffusion policy for action chunking. Aligns with model.py interface.

Model takes only (1) processed observation and (2) partially unmasked action token
sequence. No time input. Masking is specified by the input token sequence: any
position with IGNORE_TOKEN (-100) is treated as masked (use MASK embedding, predict it).
"""

import dataclasses
from typing import Literal, TypeAlias, Self

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from model import MLPMixerBlock, get_prefix_weights

PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]

# In the input action token sequence, this value means "masked" (model uses MASK embedding and predicts this position).
IGNORE_TOKEN = -100


def _train_mask_schedule(rand_time: jax.Array, method: str = "cosine") -> jax.Array:
    """Training-time mask schedule (ref.py): rand_time in [0, 1) -> mask_ratio in (0, 1].
    How much of the maskable positions to mask; used as num_mask = round(total_unknown * mask_ratio).
    """
    if method == "cosine":
        return jnp.clip(1.0 - jnp.cos(jnp.pi * 0.5 * rand_time), 1e-6, 1.0)
    elif method == "linear":
        return jnp.clip(rand_time, 1e-6, 1.0)
    else:
        raise ValueError(f"Unknown train mask schedule: {method}")


def _decode_mask_schedule(ratio: jax.Array, method: str = "cosine") -> jax.Array:
    """Decode-time schedule: ratio in [0, 1] -> mask_ratio (fraction of unknown to keep masked).
    As ratio increases (later in decode), mask_ratio decreases (more positions unmasked).
    """
    if method == "cosine":
        # ratio 0 -> 1, ratio 1 -> 0
        return 0.5 * (1.0 + jnp.cos(jnp.pi * ratio))
    elif method == "linear":
        return 1.0 - ratio
    else:
        raise ValueError(f"Unknown decode schedule: {method}")


def _mask_by_random_topk(
    rng: jax.Array,
    selected_probs: jax.Array,
    mask_len: jax.Array,
    temperature: float,
) -> jax.Array:
    """Which positions stay masked: Gumbel + top-k. selected_probs [B, L]; mask_len [B].
    Returns action_mask [B, L] with True = stay masked (low confidence / low prob).
    """
    B, L = selected_probs.shape

    gumbel_rng, perm_rng = jax.random.split(rng)
    gumbel = -jnp.log(-jnp.log(jax.random.uniform(gumbel_rng, selected_probs.shape) + 1e-10) + 1e-10)
    score = -jnp.log(selected_probs + 1e-8) / temperature + gumbel
    perm = jnp.argsort(score, axis=1)
    ranks = jnp.argsort(perm, axis=1)
    action_mask = ranks < mask_len[:, None]
    return action_mask


def continuous_to_bins(actions: jax.Array, num_bins: int) -> jax.Array:
    """Map continuous actions in [-1, 1] to bin indices in [0, num_bins-1]."""
    # Clamp to (-1, 1) then map: [-1, 1] -> [0, num_bins-1]
    clamped = jnp.clip(actions, -1.0 + 1e-6, 1.0 - 1e-6)
    bins = jnp.floor((clamped + 1.0) / 2.0 * num_bins).astype(jnp.int32)
    return jnp.clip(bins, 0, num_bins - 1)


def bins_to_continuous(bin_indices: jax.Array, num_bins: int) -> jax.Array:
    """Map bin indices to continuous actions in [-1, 1] (bin centers)."""
    # Bin center: (i + 0.5) / num_bins maps [0, num_bins) -> (0, 1], then *2-1 -> (-1, 1]
    return (jnp.asarray(bin_indices, dtype=jnp.float32) + 0.5) / num_bins * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    channel_dim: int = 256
    channel_hidden_dim: int = 512
    token_hidden_dim: int = 64
    num_layers: int = 4
    action_chunk_size: int = 8
    num_bins: int = 512
    simulated_delay: int | None = None
    # Training-time masking (ref.py): schedule for mask ratio, no_mask_token_prob, RTC first-d
    train_mask_schedule: Literal["cosine", "linear"] = "cosine"
    no_mask_token_prob: float = 0.0
    inpainting_masking: bool = False  # If True, never mask the first d steps (ref: rtc_masking)
    # Decode (realtime_decode-style): schedule and temperature
    decode_schedule: Literal["cosine", "linear"] = "cosine"
    choice_temperature: float = 1.0
    use_remask: bool = False


class DiscreteDiffusionPolicy(nnx.Module):
    """Discrete diffusion policy with interface aligned to FlowPolicy."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: ModelConfig,
        rngs: nnx.Rngs,
    ):
        self.channel_dim = config.channel_dim
        self.action_dim = action_dim
        self.action_chunk_size = config.action_chunk_size
        self.num_bins = config.num_bins
        self.simulated_delay = config.simulated_delay
        self.train_mask_schedule = config.train_mask_schedule
        self.no_mask_token_prob = config.no_mask_token_prob
        self.inpainting_masking = config.inpainting_masking
        self.decode_schedule = config.decode_schedule
        self.choice_temperature = config.choice_temperature
        self.use_remask = config.use_remask
        # Mask token index: positions with this value (or IGNORE_TOKEN in input) are "to be predicted".
        self.mask_token_id = config.num_bins

        # Input: (obs, partially unmasked action tokens). Tokens are bin indices or mask_token_id.
        # Match model.py (FlowPolicy) network size: mixer uses token_dim = action_chunk_size (8), not 8*action_dim.
        self.mixer_token_dim = config.action_chunk_size
        # num_bins + 1 to include embedding for MASK token.
        self.bin_embed = nnx.Embed(config.num_bins + 1, config.channel_dim // 2, rngs=rngs)
        self.in_proj = nnx.Linear(config.channel_dim // 2 + obs_dim, config.channel_dim, rngs=rngs)
        # Pack (chunk, action_dim, channel) -> (chunk, channel) so mixer sees action_chunk_size tokens like FlowPolicy.
        self.pack_tokens = nnx.Linear(
            action_dim * config.channel_dim, config.channel_dim, rngs=rngs
        )
        self.mlp_stack = [
            MLPMixerBlock(
                self.mixer_token_dim,
                config.token_hidden_dim,
                config.channel_dim,
                config.channel_hidden_dim,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ]
        # Learned conditioning (replaces time); broadcast to all tokens.
        self.null_cond = nnx.Param(jnp.zeros((config.channel_dim,)))
        self.final_norm = nnx.LayerNorm(config.channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.final_adaln = nnx.Linear(
            config.channel_dim, 2 * config.channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        # Unpack (chunk, channel) -> (chunk, action_dim, num_bins) to match FlowPolicy out_proj scale.
        self.out_proj = nnx.Linear(
            config.channel_dim, action_dim * config.num_bins, rngs=rngs
        )

    def __call__(self, obs: jax.Array, x_t_bins: jax.Array) -> jax.Array:
        """Predict logits for action bins given (obs, partially unmasked action sequence).

        x_t_bins: [batch, chunk, action_dim] with values in [0, num_bins-1], mask_token_id,
        or IGNORE_TOKEN (-100). IGNORE_TOKEN is treated as masked (same as mask_token_id for embedding).
        Returns [batch, chunk, action_dim, num_bins] logits over action bins (no logit for MASK).
        """
        assert x_t_bins.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_t_bins.shape
        # Map IGNORE_TOKEN (-100) to mask_token_id for embedding lookup
        x_for_embed = jnp.where(x_t_bins == IGNORE_TOKEN, self.mask_token_id, x_t_bins)
        # Embed bins (including MASK token): [batch, chunk, action_dim] -> [batch, chunk, action_dim, embed_dim]
        bin_emb = self.bin_embed(x_for_embed)
        obs_expanded = einops.repeat(obs, "b e -> b c d e", c=self.action_chunk_size, d=self.action_dim)
        x = jnp.concatenate([bin_emb, obs_expanded], axis=-1)
        x = self.in_proj(x)  # [batch, chunk, action_dim, channel_dim]
        # Pack to (batch, action_chunk_size, channel_dim) so mixer has same token_dim as model.py (FlowPolicy)
        x = x.reshape(obs.shape[0], self.action_chunk_size, self.action_dim * self.channel_dim)
        x = self.pack_tokens(x)  # [batch, action_chunk_size, channel_dim]
        batch_size = obs.shape[0]
        cond = jnp.broadcast_to(
            self.null_cond.value, (batch_size, self.mixer_token_dim, self.channel_dim)
        )
        for mlp in self.mlp_stack:
            x = mlp(x, cond)
        scale, shift = jnp.split(self.final_adaln(cond), 2, axis=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        logits = self.out_proj(x)  # [batch, action_chunk_size, action_dim * num_bins]
        logits = logits.reshape(
            obs.shape[0], self.action_chunk_size, self.action_dim, self.num_bins
        )
        return logits

    def action(self, rng: jax.Array, obs: jax.Array, num_steps: int) -> jax.Array:
        """MaskGIT-style decode from all-masked input (ref realtime_decode). Gradual unmasking via schedule + mask_by_random_topk."""
        B = obs.shape[0]
        L = self.action_chunk_size * self.action_dim
        cur_seqs = jnp.full(
            (B, self.action_chunk_size, self.action_dim),
            self.mask_token_id,
            dtype=jnp.int32,
        )
        unknown_init = jnp.full((B,), L, dtype=jnp.int32)

        def step(carry, step_idx):
            cur_seqs, rng_state = carry
            rng_state, cat_key, topk_key = jax.random.split(rng_state, 3)
            # 1) Logits and probs
            logits = self(obs, cur_seqs)
            probs = jax.nn.softmax(logits, axis=-1)
            # 2) Categorical sample over all positions
            sampled = jax.random.categorical(cat_key, logits, axis=-1)
            unknown_map = cur_seqs == self.mask_token_id
            sampled = jnp.where(unknown_map, sampled, cur_seqs)
            # 3) Ratio and mask_len for this step (at most unknown_init-1 so at least one unmasked)
            ratio = (step_idx + 1.0) / num_steps
            mask_ratio = _decode_mask_schedule(ratio, self.decode_schedule)
            mask_len = (unknown_init.astype(jnp.float32) * mask_ratio).astype(jnp.int32)
            mask_len = jnp.clip(mask_len, 1, jnp.maximum(unknown_init - 1, 0))
            mask_len = jnp.where(step_idx == num_steps - 1, 0, mask_len)
            # 4) Selected probs (confidence at sampled token)
            selected_probs = jnp.take_along_axis(probs, sampled[..., None], axis=-1).squeeze(-1)
            if self.use_remask:
                p_remask = 1.0 - ratio
                selected_probs = jnp.where(
                    unknown_map,
                    selected_probs,
                    selected_probs * p_remask,
                )
            else:
                selected_probs = jnp.where(unknown_map, selected_probs, jnp.float32("inf"))
            # 5) Which positions stay masked (Gumbel + top-k)
            temp = self.choice_temperature * (1.0 - ratio)
            selected_flat = selected_probs.reshape(B, L)
            action_mask_flat = _mask_by_random_topk(topk_key, selected_flat, mask_len, temp)
            action_mask = action_mask_flat.reshape(B, self.action_chunk_size, self.action_dim)
            # 6) Next seqs: masked positions keep mask_token_id, rest get sampled
            next_seqs = jnp.where(action_mask, self.mask_token_id, sampled)
            return (next_seqs, rng_state), None

        (cur_seqs, _), _ = jax.lax.scan(step, (cur_seqs, rng), jnp.arange(num_steps))
        # Final cur_seqs may still have some mask positions on non-last step; use sampled from last step.
        # Actually after last step we set mask_len=0 so action_mask is all False, next_seqs = sampled. So cur_seqs is full.
        return bins_to_continuous(cur_seqs, self.num_bins)

    def bid_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,
        inference_delay: int,
        prefix_attention_horizon: int,
        n_samples: int,
        bid_weak_policy: Self | None = None,
        bid_k: int | None = None,
    ) -> jax.Array:
        obs = einops.repeat(obs, "b ... -> (n b) ...", n=n_samples)
        weights = get_prefix_weights(inference_delay, prefix_attention_horizon, self.action_chunk_size, "exp")

        def backward_loss(action_chunks: jax.Array):
            error = jnp.linalg.norm(action_chunks - prev_action_chunk, axis=-1)
            return jnp.sum(error * weights[None, None, :], axis=-1)

        strong_actions = einops.rearrange(self.action(rng, obs, num_steps), "(n b) h d -> n b h d", n=n_samples)
        loss = backward_loss(strong_actions)

        if bid_weak_policy is not None or bid_k is not None:
            assert bid_weak_policy is not None and bid_k is not None, (bid_weak_policy, bid_k)
            weak_actions = einops.rearrange(
                bid_weak_policy.action(rng, obs, num_steps), "(n b) h d -> n b h d", n=n_samples
            )
            weak_loss = backward_loss(weak_actions)
            weak_idxs = jax.lax.top_k(-weak_loss.T, bid_k)[1].T
            strong_idxs = jax.lax.top_k(-loss.T, bid_k)[1].T
            a_plus = jnp.take_along_axis(strong_actions, strong_idxs[:, :, None, None], axis=0)
            a_minus = jnp.take_along_axis(weak_actions, weak_idxs[:, :, None, None], axis=0)
            forward_loss = jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_plus[None, :], axis=-1),
                axis=(1, 3),
            ) - jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_minus[None, :], axis=-1),
                axis=(1, 3),
            )
            loss += forward_loss / n_samples

        best_idxs = jnp.argmin(loss, axis=0)
        return jnp.take_along_axis(strong_actions, best_idxs[None, :, None, None], axis=0).squeeze(0)

    def realtime_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,
        inference_delay: int,
        prefix_attention_horizon: int | None = None,
        prefix_attention_schedule: PrefixAttentionSchedule = "exp",
        max_guidance_weight: float = 5.0,
    ) -> jax.Array:
        """MaskGIT-style decode with fixed prefix (ref realtime_decode). Prefix = first inference_delay steps; rest gradual unmask.
        Extra args (prefix_attention_horizon, prefix_attention_schedule, max_guidance_weight) kept for eval_dd API compatibility; unused.
        """
        B = obs.shape[0]
        L = self.action_chunk_size * self.action_dim
        prefix_bins = continuous_to_bins(prev_action_chunk, self.num_bins)
        prefix_mask = jnp.arange(self.action_chunk_size)[None, :, None] < inference_delay
        cur_seqs = jnp.where(
            prefix_mask,
            prefix_bins,
            jnp.full_like(prefix_bins, self.mask_token_id, dtype=jnp.int32),
        )
        unknown_init = jnp.full(
            (B,), (self.action_chunk_size - inference_delay) * self.action_dim, dtype=jnp.int32
        )

        def step(carry, step_idx):
            cur_seqs, rng_state = carry
            rng_state, cat_key, topk_key = jax.random.split(rng_state, 3)
            logits = self(obs, cur_seqs)
            probs = jax.nn.softmax(logits, axis=-1)
            sampled = jax.random.categorical(cat_key, logits, axis=-1)
            unknown_map = cur_seqs == self.mask_token_id
            sampled = jnp.where(prefix_mask, prefix_bins, jnp.where(unknown_map, sampled, cur_seqs))
            ratio = (step_idx + 1.0) / num_steps
            mask_ratio = _decode_mask_schedule(ratio, self.decode_schedule)
            mask_len = (unknown_init.astype(jnp.float32) * mask_ratio).astype(jnp.int32)
            mask_len = jnp.clip(mask_len, 1, jnp.maximum(unknown_init - 1, 0))
            mask_len = jnp.where(step_idx == num_steps - 1, 0, mask_len)
            selected_probs = jnp.take_along_axis(probs, sampled[..., None], axis=-1).squeeze(-1)
            if self.use_remask:
                p_remask = 1.0 - ratio
                selected_probs = jnp.where(
                    unknown_map,
                    selected_probs,
                    selected_probs * p_remask,
                )
            else:
                selected_probs = jnp.where(unknown_map, selected_probs, jnp.float32("inf"))
            temp = self.choice_temperature * (1.0 - ratio)
            selected_flat = selected_probs.reshape(B, L)
            action_mask_flat = _mask_by_random_topk(topk_key, selected_flat, mask_len, temp)
            action_mask = action_mask_flat.reshape(B, self.action_chunk_size, self.action_dim)
            next_seqs = jnp.where(
                prefix_mask, prefix_bins, jnp.where(action_mask, self.mask_token_id, sampled)
            )
            return (next_seqs, rng_state), None

        (cur_seqs, _), _ = jax.lax.scan(step, (cur_seqs, rng), jnp.arange(num_steps))
        return bins_to_continuous(cur_seqs, self.num_bins)

    def apply_mask(self, rng: jax.Array, input_tokens: jax.Array) -> jax.Array:
        """Apply ref-style masking to input_tokens. Positions selected for masking become IGNORE_TOKEN (-100).

        input_tokens: [batch, action_chunk_size, action_dim] token ids (e.g. target bins).
        Returns masked_tokens: same shape; masked positions set to IGNORE_TOKEN, rest unchanged.
        """
        B, L = input_tokens.shape[0], self.action_chunk_size * self.action_dim
        rng, time_rng, vals_rng, unmask_rng, delay_rng, rtc_rng = jax.random.split(rng, 6)
        # 1) Which positions can be masked (ref: loss_mask_full)
        if self.simulated_delay is not None:
            w = jnp.exp(jnp.arange(0, self.simulated_delay)[::-1])
            w = w / jnp.sum(w)
            delay = jax.random.choice(delay_rng, self.simulated_delay, (B,), p=w)
            prefix_mask = (jnp.arange(self.action_chunk_size)[None, :] < delay[:, None])[:, :, None]
            loss_mask_full = ~prefix_mask
        else:
            loss_mask_full = jnp.ones((B, self.action_chunk_size, self.action_dim), dtype=bool)
        total_unknown = jnp.sum(loss_mask_full.astype(jnp.float32), axis=(1, 2))
        # 2) Random time -> schedule -> mask_ratio -> num_mask (ref: steps 2–3)
        rand_time = jax.random.uniform(time_rng, (B,))
        mask_ratios = _train_mask_schedule(rand_time, self.train_mask_schedule)
        num_mask = jnp.clip(
            jnp.round(total_unknown * mask_ratios).astype(jnp.int32),
            1,
            L,
        )
        num_mask = jnp.where(total_unknown > 0, num_mask, 0)
        # 3) Top-k by random score among loss_mask_full -> masked_mask (ref: steps 4–5)
        vals = jax.random.uniform(vals_rng, (B, self.action_chunk_size, self.action_dim))
        vals = jnp.where(loss_mask_full, vals, jnp.float32("inf"))
        vals_flat = vals.reshape(B, L)
        perm = jnp.argsort(vals_flat, axis=1)
        ranks = jnp.argsort(perm, axis=1)
        masked_mask_flat = ranks < num_mask[:, None]
        masked_mask = masked_mask_flat.reshape(B, self.action_chunk_size, self.action_dim)
        # 4) Optional no_mask_token_prob (ref: step 6)
        if self.no_mask_token_prob > 0:
            prob = jax.random.uniform(unmask_rng, masked_mask.shape)
            unmask = (prob < self.no_mask_token_prob) & masked_mask
            masked_mask = masked_mask & ~unmask
        # 5) Optional inpainting_masking: never mask the first d steps (ref: rtc_masking)
        if self.simulated_delay is not None and self.inpainting_masking:
            d = jax.random.randint(rtc_rng, (), 1, self.action_chunk_size // 2 + 1)
            first_d_mask = jnp.arange(self.action_chunk_size)[None, :, None] >= d
            masked_mask = masked_mask & first_d_mask
        masked_tokens = jnp.where(masked_mask, IGNORE_TOKEN, input_tokens)
        return masked_tokens

    def loss(
        self,
        rng: jax.Array,
        obs: jax.Array,
        action: jax.Array,
        input_tokens: jax.Array | None = None,
    ) -> jax.Array:
        """Cross-entropy on masked positions only. Input-token masking is done in this method.

        By default (input_tokens is None), masking is applied via self.apply_mask(rng, target_bins)
        (ref-style: schedule, top-k, no_mask_token_prob, RTC first-d).
        If input_tokens is provided, it is used as-is (caller controls masking).
        """
        assert action.dtype == jnp.float32
        assert action.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), action.shape
        target_bins = continuous_to_bins(action, self.num_bins)

        if input_tokens is None:
            input_tokens = self.apply_mask(rng, target_bins)

        # Where input_tokens == IGNORE_TOKEN we use MASK embedding and compute loss
        loss_mask = input_tokens == IGNORE_TOKEN
        x_for_embed = jnp.where(loss_mask, self.mask_token_id, input_tokens)

        logits = self(obs, x_for_embed)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_one_hot = jax.nn.one_hot(target_bins, self.num_bins)
        ce_per_token = -jnp.sum(target_one_hot * log_probs, axis=-1)
        ce_masked = jnp.where(loss_mask, ce_per_token, 0.0)
        # Average over masked positions per sample, then mean over batch (not sum).
        num_masked_per_sample = jnp.sum(loss_mask, axis=(1, 2)) + 1e-8
        ce_sum_per_sample = jnp.sum(ce_masked, axis=(1, 2))
        loss_per_sample = ce_sum_per_sample / num_masked_per_sample
        return jnp.mean(loss_per_sample)
