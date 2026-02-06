"""Discrete diffusion policy for action chunking. Aligns with model.py interface."""

import dataclasses
import functools
from typing import Literal, TypeAlias, Self

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from model import MLPMixerBlock, get_prefix_weights

PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]


def posemb_sincos(pos: jax.Array, embedding_dim: int, min_period: float, max_period: float) -> jax.Array:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


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
    num_bins: int = 256
    simulated_delay: int | None = None


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

        # Input: flatten (chunk, action_dim) to tokens. Each token = one (chunk_idx, action_dim_idx) position.
        token_dim = config.action_chunk_size * action_dim
        self.bin_embed = nnx.Embed(config.num_bins, config.channel_dim // 2, rngs=rngs)
        self.in_proj = nnx.Linear(config.channel_dim // 2 + obs_dim, config.channel_dim, rngs=rngs)
        self.mlp_stack = [
            MLPMixerBlock(
                token_dim,
                config.token_hidden_dim,
                config.channel_dim,
                config.channel_hidden_dim,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ]
        self.time_mlp = nnx.Sequential(
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
        )
        self.final_norm = nnx.LayerNorm(config.channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.final_adaln = nnx.Linear(
            config.channel_dim, 2 * config.channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        self.out_proj = nnx.Linear(config.channel_dim, config.num_bins, rngs=rngs)

    def __call__(self, obs: jax.Array, x_t_bins: jax.Array, time: jax.Array) -> jax.Array:
        """Predict logits for x_0 given corrupted bins x_t. Returns [batch, chunk, action_dim, num_bins]."""
        assert x_t_bins.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_t_bins.shape
        if time.ndim == 1:
            time = time[:, None]
        time = jnp.broadcast_to(time, (obs.shape[0], self.action_chunk_size))
        time_emb = jax.vmap(
            functools.partial(posemb_sincos, embedding_dim=self.channel_dim, min_period=4e-3, max_period=4.0)
        )(time)
        time_emb = self.time_mlp(time_emb)
        # Embed bins: [batch, chunk, action_dim] -> [batch, chunk, action_dim, embed_dim]
        bin_emb = self.bin_embed(x_t_bins)
        obs_expanded = einops.repeat(obs, "b e -> b c d e", c=self.action_chunk_size, d=self.action_dim)
        x = jnp.concatenate([bin_emb, obs_expanded], axis=-1)
        x = self.in_proj(x)
        # Reshape to (batch, token_dim, channel_dim) for MLPMixerBlock
        x = x.reshape(obs.shape[0], self.action_chunk_size * self.action_dim, self.channel_dim)
        # Expand time_emb for token_dim (broadcast over tokens)
        time_emb = einops.repeat(time_emb, "b c e -> b (c d) e", d=self.action_dim)
        for mlp in self.mlp_stack:
            x = mlp(x, time_emb)
        scale, shift = jnp.split(self.final_adaln(time_emb), 2, axis=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        logits = self.out_proj(x)  # [batch, token_dim, num_bins]
        logits = logits.reshape(obs.shape[0], self.action_chunk_size, self.action_dim, self.num_bins)
        return logits

    def action(self, rng: jax.Array, obs: jax.Array, num_steps: int) -> jax.Array:
        """Sample action chunk via iterative denoising. Returns [batch, chunk, action_dim] continuous."""
        rng, init_rng = jax.random.split(rng)
        x_t_bins = jax.random.randint(
            init_rng, (obs.shape[0], self.action_chunk_size, self.action_dim), 0, self.num_bins
        )

        def step(carry, step_idx):
            x_t_bins, rng_state = carry
            t = (step_idx + 1) / num_steps
            rng_state, key = jax.random.split(rng_state)
            # Predict x_0 from model, use for denoising
            time_arr = jnp.full((obs.shape[0],), t - 0.5 / num_steps)  # Mid-step time
            logits = self(obs, x_t_bins, time_arr)
            # Blend: at early steps sample from pred, at late steps use argmax
            use_argmax = step_idx == num_steps - 1
            x_0_bins = jnp.argmax(logits, axis=-1)
            x_0_sampled = jax.random.categorical(key, logits, axis=-1)
            next_bins = jnp.where(use_argmax, x_0_bins, x_0_sampled)
            return (next_bins, rng_state), None

        (x_1_bins, _), _ = jax.lax.scan(
            step, (x_t_bins, rng), jnp.arange(num_steps)
        )
        return bins_to_continuous(x_1_bins, self.num_bins)

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
        prefix_attention_horizon: int,
        prefix_attention_schedule: PrefixAttentionSchedule,
        max_guidance_weight: float,
    ) -> jax.Array:
        """Realtime action with prefix masking: already-executed actions stay fixed."""
        prefix_bins = continuous_to_bins(prev_action_chunk, self.num_bins)
        prefix_mask = jnp.arange(self.action_chunk_size)[None, :, None] < inference_delay

        rng, init_rng = jax.random.split(rng)
        x_t_bins = jax.random.randint(
            init_rng, (obs.shape[0], self.action_chunk_size, self.action_dim), 0, self.num_bins
        )
        x_t_bins = jnp.where(prefix_mask, prefix_bins, x_t_bins)

        def step(carry, step_idx):
            x_t_bins, rng_state = carry
            t = (step_idx + 1) / num_steps
            rng_state, key = jax.random.split(rng_state)
            time_arr = jnp.full((obs.shape[0],), t - 0.5 / num_steps)
            logits = self(obs, x_t_bins, time_arr)
            use_argmax = step_idx == num_steps - 1
            x_0_bins = jnp.argmax(logits, axis=-1)
            x_0_sampled = jax.random.categorical(key, logits, axis=-1)
            next_bins = jnp.where(use_argmax, x_0_bins, x_0_sampled)
            # Keep prefix fixed
            next_bins = jnp.where(prefix_mask, prefix_bins, next_bins)
            return (next_bins, rng_state), None

        (x_1_bins, _), _ = jax.lax.scan(step, (x_t_bins, rng), jnp.arange(num_steps))
        return bins_to_continuous(x_1_bins, self.num_bins)

    def loss(self, rng: jax.Array, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Cross-entropy loss for discrete diffusion. Action is continuous [batch, chunk, action_dim]."""
        assert action.dtype == jnp.float32
        assert action.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), action.shape
        time_rng, corrupt_rng, noise_rng, delay_rng = jax.random.split(rng, 4)

        target_bins = continuous_to_bins(action, self.num_bins)
        time = jax.random.uniform(time_rng, (obs.shape[0],))

        # Corrupt: with probability time, replace each bin with random
        corrupt_rand = jax.random.uniform(corrupt_rng, target_bins.shape)
        random_bins = jax.random.randint(noise_rng, target_bins.shape, 0, self.num_bins)
        x_t_bins = jnp.where(corrupt_rand < time[:, None, None], random_bins, target_bins)

        if self.simulated_delay is not None:
            w = jnp.exp(jnp.arange(0, self.simulated_delay)[::-1])
            w = w / jnp.sum(w)
            delay = jax.random.choice(delay_rng, self.simulated_delay, (obs.shape[0],), p=w)
            mask = jnp.arange(self.action_chunk_size)[None, :] < delay[:, None]
            # For masked positions, x_t = target (no corruption)
            x_t_bins = jnp.where(mask[:, :, None], target_bins, x_t_bins)
            # Loss only on non-masked
            loss_mask = jnp.logical_not(mask)[:, :, None]
        else:
            loss_mask = jnp.ones_like(target_bins, dtype=bool)

        logits = self(obs, x_t_bins, time)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_one_hot = jax.nn.one_hot(target_bins, self.num_bins)
        ce_per_token = -jnp.sum(target_one_hot * log_probs, axis=-1)
        ce_masked = jnp.where(loss_mask, ce_per_token, 0.0)
        return jnp.sum(ce_masked) / (jnp.sum(loss_mask) + 1e-8)
