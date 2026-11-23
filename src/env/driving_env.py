"""JAX vehicle dynamics and route shaping for the robo-taxi arena."""

from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from env.arenas import SERVICE_HALL
from env.base import BaseEnv, BaseEnvParams, BaseEnvState, RADIUS


GRID_SPACING = 0.5
MOVES = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


class RobotaxiParams(BaseEnvParams):
    mover_centers: jax.Array
    mover_travel: jax.Array
    mover_rates: jax.Array
    wheelbase: float
    max_speed: float
    max_acceleration: float
    max_braking: float
    max_steering: float
    rolling_friction: float
    drag: float
    route_cost: jax.Array
    lookahead: jax.Array


class RobotaxiState(BaseEnvState):
    heading: jax.Array
    speed: jax.Array
    obstacle_phase: jax.Array


class RobotaxiObservation(NamedTuple):
    speed: jax.Array
    goal_local: jax.Array
    route_local: jax.Array
    goal_distance: jax.Array
    ray_distances: jax.Array
    collision_rays: jax.Array

    def as_vector(self):
        return jnp.concatenate((
            jnp.stack((self.speed, self.goal_distance), axis=-1),
            self.goal_local, self.route_local, self.ray_distances,
        ), axis=-1)


@partial(jax.jit, static_argnames=("arena_size", "preferred_clearance"))
def navigation_field(obstacles, goal, arena_size, preferred_clearance):
    """Bellman relaxation on an eight-neighbor grid of valid vehicle centers."""
    width, height = arena_size
    shape = (int(width / GRID_SPACING) + 1, int(height / GRID_SPACING) + 1)
    indices = jnp.moveaxis(jnp.indices(shape), 0, -1)
    points = indices * GRID_SPACING
    clearance = jnp.min(jnp.linalg.norm(points[..., None, :] - obstacles, axis=-1), axis=-1)
    wall_clearance = jnp.min(jnp.minimum(points, jnp.array(arena_size) - points), axis=-1)
    valid = (clearance >= 2 * RADIUS + 0.1) & (wall_clearance >= RADIUS + 0.1)
    goal_cell = jnp.rint(goal / GRID_SPACING).astype(jnp.int32)
    costs = jnp.full(shape, jnp.inf).at[goal_cell[0], goal_cell[1]].set(0.0)

    foot_cost = 1 + 1.5 * jnp.maximum(0.0, preferred_clearance - clearance)
    direct_costs = jnp.array(
        [GRID_SPACING * (dx * dx + dy * dy) ** 0.5 for dx, dy in MOVES]
    )

    def neighbors(grid, fill):
        padded = jnp.pad(grid, 1, constant_values=fill)
        return jnp.stack([
            padded[1 + dx:1 + dx + shape[0], 1 + dy:1 + dy + shape[1]]
            for dx, dy in MOVES
        ], axis=-1)

    # A diagonal must have both neighboring cardinal cells free.
    cardinal = neighbors(valid, False)
    diagonal_clear = jnp.stack(
        [cardinal[..., (i - 1) % 8] & cardinal[..., (i + 1) % 8]
         if i % 2 else jnp.ones(shape, dtype=bool) for i in range(8)],
        axis=-1,
    )

    def candidate_costs(grid):
        return jnp.where(
            diagonal_clear, neighbors(grid, jnp.inf) + foot_cost[..., None] * direct_costs, jnp.inf
        )

    def relax(_, grid):
        return jnp.where(valid, jnp.minimum(grid, jnp.min(candidate_costs(grid), axis=-1)), jnp.inf)

    costs = jax.lax.fori_loop(0, sum(shape), relax, costs)
    next_move = jnp.array(MOVES)[jnp.argmin(candidate_costs(costs), axis=-1)]
    next_cell = jnp.where((jnp.isfinite(costs) & (costs > 0))[..., None], indices + next_move, indices)

    def advance(_, cells):
        return next_cell[cells[..., 0], cells[..., 1]]

    # Store four-grid-step lookahead targets and discourage moving into invalid cells.
    lookahead = jax.lax.fori_loop(0, 4, advance, indices) * GRID_SPACING
    finite_cost = jnp.max(jnp.where(jnp.isfinite(costs), costs, 0.0))
    return jnp.where(jnp.isfinite(costs), costs, finite_cost + 2.0), lookahead, next_cell


def interpolate(grid, position):
    cell = jnp.clip(position / GRID_SPACING, 0, jnp.array(grid.shape[:2]) - 1.0001)
    lower = jnp.floor(cell).astype(jnp.int32)
    x, y = lower
    u, v = cell - lower
    return (
        (1 - u) * (1 - v) * grid[x, y] + u * (1 - v) * grid[x + 1, y]
        + (1 - u) * v * grid[x, y + 1] + u * v * grid[x + 1, y + 1]
    )


def moving_obstacles(phase, params):
    positions = params.mover_centers + jnp.sin(phase)[:, None] * params.mover_travel
    velocities = (params.mover_rates * jnp.cos(phase))[:, None] * params.mover_travel
    return positions, velocities


class RobotaxiEnv(BaseEnv):
    """A forward-only bicycle model with static and moving unit circles."""

    @staticmethod
    def init_params(key, max_steps=None, fps=10, num_ray_sensors=24, perception_radius=6.0):
        arena = SERVICE_HALL
        width, height = arena.size
        start = jnp.array(arena.start)
        goal = jnp.array(arena.goal)
        obstacles = jnp.array(arena.static_centers())
        route_cost, lookahead, next_cells = navigation_field(
            obstacles, goal, arena.size, arena.preferred_clearance
        )
        start_cell = jnp.rint(start / GRID_SPACING).astype(jnp.int32)

        def follow_route(cell, _):
            next_cell = next_cells[cell[0], cell[1]]
            return next_cell, next_cell * GRID_SPACING

        path_length = sum(route_cost.shape)
        _, path = jax.lax.scan(follow_route, start_cell, None, length=path_length)
        centers, travel, rates = zip(*arena.movers, strict=True)
        params = RobotaxiParams(
            max_steps_in_episode=arena.max_steps if max_steps is None else max_steps,
            fps=fps, step_size=1 / fps,
            mover_centers=jnp.array(centers),
            mover_travel=jnp.array(travel), mover_rates=jnp.array(rates),
            map_height_width=(height, width),
            nav_grid_shape_dtype=(route_cost.shape, jnp.float32),
            path_array_shape_dtype=(path.shape, jnp.float32),
            discretization_scale=2, num_ray_sensors=num_ray_sensors,
            agent_FOV=float(jnp.pi), perception_radius=perception_radius,
            wheelbase=1.35, max_speed=4.0, max_acceleration=4.0, max_braking=6.0,
            max_steering=0.64, rolling_friction=0.24, drag=0.12,
            route_cost=route_cost, lookahead=lookahead,
        )
        phase = jnp.zeros(len(arena.movers))
        positions, velocities = moving_obstacles(phase, params)
        initial_state = RobotaxiState(
            time=jnp.asarray(0), goal_pos=goal, agent_pos=start,
            agent_forward_dir=jnp.array([1.0, 0.0]), static_obstacles=obstacles,
            kinematic_obstacles=positions, kinematic_obst_velocities=velocities,
            path_array=path, heading=jnp.asarray(0.0), speed=jnp.asarray(0.0),
            obstacle_phase=phase,
        )
        return params, initial_state

    @staticmethod
    @eqx.filter_jit
    def reset(key, env_params, init_state, randomize=True):
        state = init_state
        if randomize:
            position_key, angle_key, phase_key = jax.random.split(key, 3)
            position = state.agent_pos + jax.random.uniform(
                position_key, (2,), minval=-0.45, maxval=0.45
            )
            heading = jax.random.uniform(angle_key, minval=-0.3, maxval=0.3)
            phase = jax.random.uniform(
                phase_key, state.obstacle_phase.shape, minval=-jnp.pi, maxval=jnp.pi
            )
            positions, velocities = moving_obstacles(phase, env_params)
            state = eqx.tree_at(
                lambda s: (s.agent_pos, s.heading, s.agent_forward_dir, s.obstacle_phase,
                           s.kinematic_obstacles, s.kinematic_obst_velocities),
                state,
                (position, heading, jnp.stack((jnp.cos(heading), jnp.sin(heading))),
                 phase, positions, velocities),
            )
        return RobotaxiEnv.get_observation(state, env_params), state

    @staticmethod
    def get_observation(env_state, env_params):
        position = env_state.agent_pos
        forward = env_state.agent_forward_dir
        left = jnp.stack((-forward[1], forward[0]))

        def local_unit(vector):
            unit = vector / jnp.maximum(jnp.linalg.norm(vector), 1e-6)
            return jnp.stack((jnp.dot(unit, forward), jnp.dot(unit, left)))

        goal = env_state.goal_pos - position
        route = interpolate(env_params.lookahead, position) - position
        route = jnp.where(jnp.linalg.norm(goal) < 2, goal, route)
        obstacles = jnp.concatenate((env_state.static_obstacles, env_state.kinematic_obstacles))
        rays, distances = BaseEnv._collision_ray_intersections(position, forward, obstacles, env_params)

        # Include arena edges in the same ray observation as the circles.
        limits = jnp.array(env_params.map_height_width[::-1])
        offset = jnp.where(rays > 0, limits - position, -position)
        to_edge = jnp.where(jnp.abs(rays) > 1e-6, offset / rays, jnp.inf)
        distances = jnp.clip(jnp.minimum(distances, jnp.min(to_edge, axis=-1)), 0, env_params.perception_radius)
        return RobotaxiObservation(
            env_state.speed / env_params.max_speed, local_unit(goal), local_unit(route),
            jnp.linalg.norm(goal) / jnp.linalg.norm(limits), distances / env_params.perception_radius,
            rays * distances[:, None],
        )

    @staticmethod
    @eqx.filter_jit
    def step(key, env_state, action, env_params):
        state, params = env_state, env_params
        dt = params.step_size
        throttle, steering = jnp.clip(action, -1.0, 1.0)
        acceleration = throttle * jnp.where(throttle >= 0, params.max_acceleration, params.max_braking)
        resistance = params.rolling_friction + params.drag * state.speed ** 2
        speed = jnp.clip(state.speed + dt * (acceleration - resistance), 0, params.max_speed)
        mid_speed = 0.5 * (state.speed + speed)
        turn = dt * mid_speed / params.wheelbase * jnp.tan(steering * params.max_steering)
        heading = (state.heading + turn + jnp.pi) % (2 * jnp.pi) - jnp.pi
        mid_heading = state.heading + 0.5 * turn
        position = state.agent_pos + dt * mid_speed * jnp.stack(
            (jnp.cos(mid_heading), jnp.sin(mid_heading))
        )
        phase = state.obstacle_phase + params.mover_rates * dt
        moving, velocities = moving_obstacles(phase, params)
        forward = jnp.stack((jnp.cos(heading), jnp.sin(heading)))
        time = state.time + 1
        new_state = eqx.tree_at(
            lambda s: (s.time, s.agent_pos, s.agent_forward_dir, s.speed, s.heading,
                       s.obstacle_phase, s.kinematic_obstacles, s.kinematic_obst_velocities),
            state,
            (time, position, forward, speed, heading, phase, moving, velocities),
        )

        obstacles_before = jnp.concatenate((state.static_obstacles, state.kinematic_obstacles))
        obstacles_after = jnp.concatenate((state.static_obstacles, moving))
        before = state.agent_pos - obstacles_before
        after = position - obstacles_after
        relative_motion = after - before
        closest = jnp.clip(
            -jnp.sum(before * relative_motion, axis=-1) /
            jnp.maximum(jnp.sum(relative_motion ** 2, axis=-1), 1e-8), 0.0, 1.0
        )
        distance_squared = jnp.sum((before + closest[:, None] * relative_motion) ** 2, axis=-1)
        outside = jnp.any((position < RADIUS) | (position > jnp.array(params.map_height_width[::-1]) - RADIUS))
        collision = outside | jnp.any(distance_squared < (2 * RADIUS) ** 2)
        success = BaseEnv._check_goal(position, state.goal_pos) & ~collision
        timeout = (time >= params.max_steps_in_episode) & ~(success | collision)
        progress = interpolate(params.route_cost, state.agent_pos) - interpolate(params.route_cost, position)
        reward = 1.4 * jnp.clip(progress, -0.5, 0.5) - 0.025 - 0.015 * steering ** 2
        reward = jnp.where(collision, -10.0, reward + 12.0 * success - 3.0 * timeout)

        info = {"success": success, "collision": collision, "timeout": timeout}
        observation = RobotaxiEnv.get_observation(new_state, params)
        return observation, new_state, reward, success | collision | timeout, info
