"""A scripted steering controller for the environment demo and comparison."""

import equinox as eqx
import jax.numpy as jnp


@eqx.filter_jit
def route_follower(obs, params):
    heading_error = jnp.arctan2(obs.route_local[1], obs.route_local[0])
    steering = jnp.clip(1.5 * heading_error, -1, 1)
    # Slow down for corners and objects in the forward part of the ray fan.
    middle = obs.ray_distances.shape[0] // 2
    clearance = jnp.min(obs.ray_distances[middle - 1:middle + 2]) * params.perception_radius
    corner_speed = 2.9 - 1.3 * jnp.minimum(jnp.abs(heading_error), 1.0)
    obstacle_speed = jnp.clip(1.8 * (clearance - 1.1), 0, 3.0)
    target_speed = jnp.minimum(corner_speed, obstacle_speed)
    speed = obs.speed * params.max_speed
    acceleration = 2 * (target_speed - speed) + params.rolling_friction + params.drag * speed ** 2
    throttle = acceleration / params.max_acceleration
    return jnp.stack((jnp.clip(throttle, -1, 1), steering))
