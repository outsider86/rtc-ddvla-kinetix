"""Evaluation script for Discrete Diffusion policies. Mirrors eval_flow.py with multi-GPU sharding."""

import os
if "EVAL_DD_GPU_ID" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["EVAL_DD_GPU_ID"]

import collections
import dataclasses
import functools
import math
import pathlib
import pickle
import subprocess
import sys
import tempfile
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv                       # type: ignore
import kinetix.environment.env_state as kenv_state           # type: ignore
import kinetix.environment.wrappers as wrappers              # type: ignore
import kinetix.render.renderer_pixels as renderer_pixels     # type: ignore
import pandas as pd
import tyro

import model_dd as _model_dd
import train_expert


@dataclasses.dataclass(frozen=True)
class NaiveMethodConfig:
    pass


@dataclasses.dataclass(frozen=True)
class RealtimeMethodConfig:
    prefix_attention_schedule: _model_dd.PrefixAttentionSchedule = "exp"
    max_guidance_weight: float = 5.0


@dataclasses.dataclass(frozen=True)
class BIDMethodConfig:
    n_samples: int = 16
    bid_k: int | None = None


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1
    weak_step: int | None = None
    num_evals: int = 2048
    num_flow_steps: int = 5

    inference_delay: int = 0
    execute_horizon: int = 1
    method: NaiveMethodConfig | RealtimeMethodConfig | BIDMethodConfig = NaiveMethodConfig()

    model: _model_dd.ModelConfig = _model_dd.ModelConfig()


def eval(
    config: EvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    policy: _model_dd.DiscreteDiffusionPolicy,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
    weak_policy: _model_dd.DiscreteDiffusionPolicy | None = None,
):
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), config.num_evals
    )
    render_video = train_expert.make_render_video(renderer_pixels.make_render_pixels(env_params, static_env_params))
    assert config.execute_horizon >= config.inference_delay, f"{config.execute_horizon=} {config.inference_delay=}"

    def execute_chunk(carry, _):
        def step(carry, action):
            rng, obs, env_state = carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            return (rng, next_obs, next_env_state), (done, env_state, info)

        rng, obs, env_state, action_chunk, n = carry
        rng, key = jax.random.split(rng)
        if isinstance(config.method, NaiveMethodConfig):
            next_action_chunk = policy.action(key, obs, config.num_flow_steps)
        elif isinstance(config.method, RealtimeMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            next_action_chunk = policy.realtime_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.prefix_attention_schedule,
                config.method.max_guidance_weight,
            )
        elif isinstance(config.method, BIDMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            if config.method.bid_k is not None:
                assert weak_policy is not None, "weak_policy is required for BID"
            next_action_chunk = policy.bid_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.n_samples,
                bid_k=config.method.bid_k,
                bid_weak_policy=weak_policy if config.method.bid_k is not None else None,
            )
        else:
            raise ValueError(f"Unknown method: {config.method}")

        action_chunk_to_execute = jnp.concatenate(
            [
                action_chunk[:, : config.inference_delay],
                next_action_chunk[:, config.inference_delay : config.execute_horizon],
            ],
            axis=1,
        )
        next_action_chunk = jnp.concatenate(
            [
                next_action_chunk[:, config.execute_horizon :],
                jnp.zeros((obs.shape[0], config.execute_horizon, policy.action_dim)),
            ],
            axis=1,
        )
        next_n = jnp.concatenate([n[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])
        (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step, (rng, obs, env_state), action_chunk_to_execute.transpose(1, 0, 2)
        )
        return (rng, next_obs, next_env_state, next_action_chunk, next_n), (dones, env_states, infos)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)
    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n),
        None,
        length=scan_length,
    )
    dones, env_states, infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (dones, env_states, infos))
    assert dones.shape[0] >= env_params.max_timesteps, f"{dones.shape=}"
    return_info = {}
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        first_done_idx = jnp.argmax(dones, axis=0)
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    for key in ["match"]:
        if key in infos:
            return_info[key] = jnp.mean(infos[key])
    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    return return_info, video


def run_eval_chunk(
    run_path: str,
    config: EvalConfig,
    level_paths: Sequence[str],
    seed: int,
) -> list[dict]:
    """Run full eval sweep for a subset of levels; returns list of row dicts (one per level × delay × method × horizon)."""
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    state_dicts = []
    weak_state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        with (log_dirs[config.step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
        if config.weak_step is not None:
            with (log_dirs[config.weak_step] / "policies" / f"{level_name}.pkl").open("rb") as f:
                weak_state_dicts.append(pickle.load(f))
    if config.weak_step is None:
        weak_state_dicts = None

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]
    action_chunk_size = config.model.action_chunk_size

    @functools.partial(jax.jit, static_argnums=(0,))
    def _eval_single(
        config: EvalConfig,
        rng: jax.Array,
        level: kenv_state.EnvState,
        state_dict: dict,
        weak_state_dict: dict | None,
    ):
        policy = _model_dd.DiscreteDiffusionPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config.model,
            rngs=nnx.Rngs(rng),
        )
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        if weak_state_dict is not None:
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(weak_state_dict)
            weak_policy = nnx.merge(graphdef, state)
        else:
            weak_policy = None
        eval_info, _ = eval(config, env, rng, level, policy, env_params, static_env_params, weak_policy)
        return eval_info

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))

    def run_config(c: EvalConfig) -> list[dict]:
        out_list = []
        for i in range(len(level_paths)):
            level_i = jax.tree.map(lambda x: x[i], levels)
            w = weak_state_dicts[i] if weak_state_dicts is not None else None
            info = _eval_single(c, rngs[i], level_i, state_dicts[i], w)
            out_list.append(jax.device_get(info))
        return out_list

    rows: list[dict] = []
    for inference_delay in [0, 1, 2, 3, 4]:
        for execute_horizon in range(max(1, inference_delay), action_chunk_size - inference_delay + 1):
            print(f"{inference_delay=} {execute_horizon=}")
            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig()
            )
            out_list = run_config(c)
            for i in range(len(level_paths)):
                row = {"delay": inference_delay, "method": "naive", "level": level_paths[i], "execute_horizon": execute_horizon, **out_list[i]}
                rows.append(row)

            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=RealtimeMethodConfig()
            )
            out_list = run_config(c)
            for i in range(len(level_paths)):
                row = {"delay": inference_delay, "method": "realtime", "level": level_paths[i], "execute_horizon": execute_horizon, **out_list[i]}
                rows.append(row)

            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=BIDMethodConfig()
            )
            out_list = run_config(c)
            for i in range(len(level_paths)):
                row = {"delay": inference_delay, "method": "bid", "level": level_paths[i], "execute_horizon": execute_horizon, **out_list[i]}
                rows.append(row)

            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=RealtimeMethodConfig(prefix_attention_schedule="zeros"),
            )
            out_list = run_config(c)
            for i in range(len(level_paths)):
                row = {"delay": inference_delay, "method": "hard_masking", "level": level_paths[i], "execute_horizon": execute_horizon, **out_list[i]}
                rows.append(row)
    return rows


def main(
    run_path: str,
    config: EvalConfig = EvalConfig(),
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    ),
    seed: int = 0,
    output_dir: str | None = "eval_output",
    num_gpus: int = 0,
):
    """Run evaluation. Use num_gpus > 1 to distribute levels across GPUs via separate processes (avoids NCCL)."""
    level_paths = list(level_paths)
    if num_gpus <= 1:
        rows = run_eval_chunk(run_path, config, level_paths, seed)
    else:
        num_gpus = min(num_gpus, len(level_paths))
        print(f"Using {num_gpus} GPUs (one process per GPU)")
        chunk_size = (len(level_paths) + num_gpus - 1) // num_gpus
        chunks = [level_paths[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus)]
        chunks = [c for c in chunks if c]
        project_root = pathlib.Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "eval_dd.py"
        all_rows: list[dict] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            procs = []
            out_paths = []
            for i, chunk in enumerate(chunks):
                in_path = pathlib.Path(tmpdir) / f"in_{i}.pkl"
                out_paths.append(pathlib.Path(tmpdir) / f"out_{i}.pkl")
                with in_path.open("wb") as f:
                    pickle.dump((run_path, config, chunk, seed + i), f)
                env = {**os.environ, "EVAL_DD_GPU_ID": str(i)}
                procs.append(
                    subprocess.Popen(
                        [sys.executable, str(script_path), "worker", str(in_path), str(out_paths[-1])],
                        cwd=project_root,
                        env=env,
                    )
                )
            for p in procs:
                p.wait()
                if p.returncode != 0:
                    raise RuntimeError(f"Worker process exited with code {p.returncode}")
            for out_path in out_paths:
                with out_path.open("rb") as f:
                    all_rows.extend(pickle.load(f))
        rows = all_rows

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(pathlib.Path(output_dir) / "results.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "worker":
        in_path, out_path = sys.argv[2], sys.argv[3]
        with open(in_path, "rb") as f:
            run_path, config, level_paths, seed = pickle.load(f)
        rows = run_eval_chunk(run_path, config, level_paths, seed)
        with open(out_path, "wb") as f:
            pickle.dump(rows, f)
    else:
        tyro.cli(main)
