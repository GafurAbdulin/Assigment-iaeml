"""Train PPO and measure goal completion on randomized initial conditions."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorithms.ppo import initialize, load_model, make_update, predict, save_model
from algorithms.route import route_follower
from env.driving_env import RobotaxiEnv


def evaluator(params, initial, action_fn):
    reset_batch = jax.vmap(RobotaxiEnv.reset, in_axes=(0, None, None))
    step_batch = jax.vmap(RobotaxiEnv.step, in_axes=(None, 0, 0, None))

    @jax.jit
    def run(model, seeds):
        initial_keys = jax.vmap(jax.random.PRNGKey)(seeds)
        obs, state = reset_batch(initial_keys, params, initial)
        zeros = jnp.zeros(seeds.shape)
        alive = jnp.ones(seeds.shape, dtype=jnp.bool_)

        def step(carry, index):
            obs, state, alive, returns, success, collisions, lengths = carry
            action_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(initial_keys, index)
            actions = action_fn(model, obs, action_keys)
            obs, state, reward, done, info = step_batch(initial_keys[0], state, actions, params)
            return (
                obs, state, alive & ~done,
                returns + jnp.where(alive, reward, 0),
                success + (alive & info["success"]),
                collisions + (alive & info["collision"]),
                lengths + alive,
            ), None

        result, _ = jax.lax.scan(
            step, (obs, state, alive, zeros, zeros, zeros, zeros),
            jnp.arange(params.max_steps_in_episode),
        )
        return jnp.stack((result[3], result[4], result[5], result[6]), axis=1)

    return run


def policy_actions(model, observations, keys):
    mean, _, _ = predict(model, observations.as_vector())
    return jnp.tanh(mean)


def random_actions(model, observations, keys):
    return jax.vmap(lambda key: jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0))(keys)


def print_evaluation(label, results):
    returns, successes, crashes, steps = np.asarray(results).T
    count = len(returns)
    goals, collisions = int(successes.sum()), int(crashes.sum())
    print(f"{label}: goals {goals}/{count}, collisions {collisions}/{count}, "
          f"timeouts {count-goals-collisions}/{count}, mean return {returns.mean():.2f}, "
          f"mean steps {steps.mean():.1f}", flush=True)


def train(args):
    rng = jax.random.PRNGKey(args.seed)
    rng, init_key, model_key, env_key = jax.random.split(rng, 4)
    env = RobotaxiEnv
    params, initial = env.init_params(env_key)
    reset_batch = jax.vmap(env.reset, in_axes=(0, None, None))
    obs, state = reset_batch(jax.random.split(init_key, args.envs), params, initial)
    obs = obs.as_vector()
    model = initialize(model_key, obs.shape[-1])
    schedule = optax.linear_schedule(args.learning_rate, 0.00005,
                                     args.updates * args.epochs * args.minibatches)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(schedule, eps=1e-5))
    optimizer_state = optimizer.init(model)
    update = make_update(env, params, initial, optimizer, args.envs,
                         args.horizon, args.epochs, args.minibatches)
    evaluate = evaluator(params, initial, policy_actions)
    validation_seeds = jnp.arange(100_000, 100_064)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "policy.npz"
    initial_results = evaluate(model, validation_seeds)
    print_evaluation("Untrained validation", initial_results)
    initial_metrics = np.asarray(initial_results)
    best = (int(initial_metrics[:, 1].sum()), float(initial_metrics[:, 0].mean()))
    save_model(checkpoint, model)
    started = perf_counter()
    recent = np.zeros(3)

    with (args.output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        metrics = csv.writer(stream)
        metrics.writerow(["update", "environment_steps", "seconds", "loss",
                          "training_episodes", "training_successes", "training_collisions",
                          "validation_mean_return", "validation_successes", "validation_collisions"])
        for index in range(1, args.updates + 1):
            model, optimizer_state, rng, obs, state, stats = update(
                model, optimizer_state, rng, obs, state
            )
            loss, *counts = np.asarray(jnp.stack(stats))
            recent += counts
            if index == 1 or index % args.eval_every == 0 or index == args.updates:
                results = np.asarray(evaluate(model, validation_seeds))
                mean_return = float(results[:, 0].mean())
                goals, collisions = results[:, 1:3].sum(axis=0).astype(int)
                score = (goals, mean_return)
                if score > best:
                    best = score
                    save_model(checkpoint, model)
                elapsed = perf_counter() - started
                steps = index * args.envs * args.horizon
                print(f"Update {index:3d} | steps {steps:>9,d} | "
                      f"validation {goals}/64 goals, {collisions}/64 collisions | "
                      f"return {mean_return:6.2f} | {elapsed:.1f}s", flush=True)
                metrics.writerow([index, steps, round(elapsed, 2), float(loss), *recent.astype(int),
                                  mean_return, goals, collisions])
                stream.flush()
                recent.fill(0)
    print(f"Saved {checkpoint}", flush=True)


def assess(args):
    params, initial = RobotaxiEnv.init_params(jax.random.PRNGKey(0))
    seeds = jnp.arange(args.first_seed, args.first_seed + args.episodes)
    trained = load_model(args.checkpoint)
    random_results = evaluator(params, initial, random_actions)(None, seeds)
    script_actions = jax.vmap(route_follower, in_axes=(0, None))
    route_results = evaluator(params, initial, lambda _, obs, keys: script_actions(obs, params))(None, seeds)
    ppo_results = evaluator(params, initial, policy_actions)(trained, seeds)
    print_evaluation("Random actions", random_results)
    print_evaluation("Scripted route follower", route_results)
    print_evaluation("Trained PPO", ppo_results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["controller", "seed", "return", "success", "collision", "steps"])
            for label, results in [("random", random_results), ("route", route_results), ("ppo", ppo_results)]:
                for seed, (reward, success, collision, steps) in zip(np.asarray(seeds), np.asarray(results)):
                    writer.writerow([label, int(seed), float(reward), int(success), int(collision), int(steps)])


def main():
    artifacts = Path("artifacts/room")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--updates", type=int, default=480)
    training.add_argument("--envs", type=int, default=64)
    training.add_argument("--horizon", type=int, default=96)
    training.add_argument("--epochs", type=int, default=4)
    training.add_argument("--minibatches", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=0.0003)
    training.add_argument("--eval-every", type=int, default=20)
    training.add_argument("--output", type=Path, default=artifacts / "fresh-training")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, default=artifacts / "policy.npz")
    evaluation.add_argument("--episodes", type=int, default=128)
    evaluation.add_argument("--first-seed", type=int, default=200_000)
    evaluation.add_argument("--output", type=Path, default=artifacts / "evaluation.csv")
    args = parser.parse_args()
    if args.command == "train":
        if min(args.updates, args.envs, args.horizon, args.epochs, args.minibatches, args.eval_every) < 1:
            parser.error("training counts must be positive")
        train(args)
    else:
        if args.episodes < 1:
            parser.error("episodes must be positive")
        assess(args)


if __name__ == "__main__":
    main()
