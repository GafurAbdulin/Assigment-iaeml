"""Continuous-action PPO with batched JAX simulation and updates."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax


HIDDEN_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2


class ActorCritic(NamedTuple):
    input_weights: jax.Array
    input_bias: jax.Array
    hidden_weights: jax.Array
    hidden_bias: jax.Array
    actor_weights: jax.Array
    actor_bias: jax.Array
    critic_weights: jax.Array
    critic_bias: jax.Array
    log_std: jax.Array


class Transition(NamedTuple):
    obs: jax.Array
    raw_action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    success: jax.Array
    collision: jax.Array


def initialize(key, obs_size):
    keys = jax.random.split(key, 4)

    def layer(key, width_in, width_out, gain):
        weights = jax.nn.initializers.orthogonal(gain)(key, (width_in, width_out))
        return weights, jnp.zeros(width_out)

    return ActorCritic(
        *layer(keys[0], obs_size, HIDDEN_SIZE, 2 ** 0.5),
        *layer(keys[1], HIDDEN_SIZE, HIDDEN_SIZE, 2 ** 0.5),
        *layer(keys[2], HIDDEN_SIZE, 2, 0.01),
        *layer(keys[3], HIDDEN_SIZE, 1, 1.0),
        jnp.array([-0.5, -0.5]),
    )


def predict(model, observation):
    features = jnp.tanh(observation @ model.input_weights + model.input_bias)
    features = jnp.tanh(features @ model.hidden_weights + model.hidden_bias)
    mean = features @ model.actor_weights + model.actor_bias
    value = (features @ model.critic_weights + model.critic_bias)[..., 0]
    return mean, jnp.clip(model.log_std, -3, 0.5), value


def log_prob(raw_action, mean, log_std):
    normal = -0.5 * (((raw_action - mean) * jnp.exp(-log_std)) ** 2 + jnp.log(2 * jnp.pi))
    # The tanh Jacobian cancels in a probability ratio for the same raw action.
    return jnp.sum(normal - log_std, axis=-1)


def advantages(batch, bootstrap_value):
    def scan_step(carry, step):
        next_value, next_advantage = carry
        alive = 1.0 - step.done.astype(jnp.float32)
        delta = step.reward + GAMMA * alive * next_value - step.value
        advantage = delta + GAMMA * GAE_LAMBDA * alive * next_advantage
        return (step.value, advantage), advantage

    _, advantage = jax.lax.scan(
        scan_step, (bootstrap_value, jnp.zeros_like(bootstrap_value)), batch, reverse=True
    )
    return advantage, advantage + batch.value


def make_update(env, env_params, initial_state, optimizer, env_count, horizon, epochs, minibatches):
    batch_size = env_count * horizon
    if batch_size % minibatches:
        raise ValueError("environments * horizon must be divisible by minibatches")

    reset_batch = jax.vmap(env.reset, in_axes=(0, None, None))
    step_batch = jax.vmap(env.step, in_axes=(None, 0, 0, None))

    def ppo_loss(model, batch, adv, returns):
        mean, log_std, value = predict(model, batch.obs)
        ratio = jnp.exp(log_prob(batch.raw_action, mean, log_std) - batch.log_prob)
        clipped_ratio = jnp.clip(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO)
        policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clipped_ratio * adv))
        value_loss = 0.5 * jnp.mean((value - returns) ** 2)
        # Gaussian entropy encourages exploration before squashing.
        entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
        return policy_loss + 0.5 * value_loss - 0.005 * entropy

    @jax.jit
    def update(model, optimizer_state, rng, obs, state):
        def roll_step(carry, _):
            rng, obs, state = carry
            rng, noise_key, reset_key = jax.random.split(rng, 3)
            mean, log_std, value = predict(model, obs)
            raw_action = mean + jnp.exp(log_std) * jax.random.normal(noise_key, mean.shape)
            next_obs, next_state, reward, done, info = step_batch(
                rng, state, jnp.tanh(raw_action), env_params
            )
            reset_obs, reset_state = reset_batch(
                jax.random.split(reset_key, env_count), env_params, initial_state
            )

            def reset_done(new, initial):
                mask = done.reshape((env_count,) + (1,) * (new.ndim - 1))
                return jnp.where(mask, initial, new)

            state = jax.tree.map(reset_done, next_state, reset_state)
            obs_next = reset_done(next_obs.as_vector(), reset_obs.as_vector())
            transition = Transition(
                obs, raw_action, log_prob(raw_action, mean, log_std), value, reward, done,
                info["success"], info["collision"],
            )
            return (rng, obs_next, state), transition

        (rng, obs, state), batch = jax.lax.scan(roll_step, (rng, obs, state), None, length=horizon)
        _, _, bootstrap = predict(model, obs)
        adv, returns = advantages(batch, bootstrap)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = jax.tree.map(
            lambda array: array.reshape((batch_size,) + array.shape[2:]), (batch, adv, returns)
        )

        def train_epoch(carry, _):
            model, optimizer_state, rng = carry
            rng, perm_key = jax.random.split(rng)
            order = jax.random.permutation(perm_key, batch_size).reshape((minibatches, -1))

            def train_minibatch(carry, indices):
                model, optimizer_state = carry
                mini, mini_adv, mini_returns = jax.tree.map(lambda array: array[indices], flat)
                loss, gradient = jax.value_and_grad(ppo_loss)(model, mini, mini_adv, mini_returns)
                updates, optimizer_state = optimizer.update(gradient, optimizer_state, model)
                return (optax.apply_updates(model, updates), optimizer_state), loss

            (model, optimizer_state), losses = jax.lax.scan(train_minibatch, (model, optimizer_state), order)
            return (model, optimizer_state, rng), losses.mean()

        (model, optimizer_state, rng), loss = jax.lax.scan(
            train_epoch, (model, optimizer_state, rng), None, length=epochs
        )
        stats = (loss.mean(), batch.done.sum(), batch.success.sum(), batch.collision.sum())
        return model, optimizer_state, rng, obs, state, stats

    return update


def save_model(path, model):
    np.savez_compressed(path, **{name: np.asarray(array)
                                for name, array in zip(ActorCritic._fields, model, strict=True)})


def load_model(path):
    with np.load(path, allow_pickle=False) as arrays:
        return ActorCritic(*(jnp.asarray(arrays[name]) for name in ActorCritic._fields))
