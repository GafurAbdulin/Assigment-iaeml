"""Drive the taxi manually, replay a controller, or record an MP4."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import pygame

from algorithms.ppo import load_model, predict
from algorithms.route import route_follower
from env.arenas import SERVICE_HALL
from env.base import RADIUS
from env.driving_env import RobotaxiEnv


def keyboard_action():
    keys = pygame.key.get_pressed()
    accelerate = keys[pygame.K_UP] or keys[pygame.K_w]
    brake = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    return jnp.array([int(accelerate) - int(brake), int(left) - int(right)], dtype=jnp.float32)


class RobotaxiFrontend:
    """Pygame frontend for the same world coordinates used by the simulation."""

    def __init__(self, params, visible):
        self.margin = 24
        height, width = params.map_height_width
        self.scale = min(32, 1056 // width, 560 // height)
        self.width = width * self.scale + 2 * self.margin
        self.field_height = height * self.scale
        self.top = 78
        size = (self.width, self.field_height + 160)
        pygame.font.init()
        if visible:
            pygame.display.init()
            pygame.display.set_caption("Assigment")
            self.surface = pygame.display.set_mode(size)
        else:
            self.surface = pygame.Surface(size)
        self.heading = pygame.font.SysFont("Segoe UI", 27, bold=True)
        self.text = pygame.font.SysFont("Segoe UI", 17)
        self.trail = []
        self.distance = 0.0
        self.last_position = None

    def reset_trail(self):
        self.trail.clear()
        self.distance = 0.0
        self.last_position = None

    def world_to_pixels(self, point):
        x, y = point
        return (round(self.margin + x * self.scale), round(self.top + self.field_height - y * self.scale))

    def label(self, value, x, y, color=(209, 222, 236), title=False):
        font = self.heading if title else self.text
        self.surface.blit(font.render(value, True, color), (x, y))

    def draw(self, state, obs, reward, status, mode, seed, params):
        self.surface.fill((12, 20, 31))
        self.label("Assigment", self.margin, 13, title=True)
        self.label(f"{mode}   |   Seed {seed}", self.margin, 48)
        field = pygame.Rect(self.margin, self.top, self.width - 2 * self.margin, self.field_height)
        pygame.draw.rect(self.surface, (21, 37, 52), field)

        for x in range(field.left, field.right, self.scale):
            pygame.draw.line(self.surface, (26, 45, 61), (x, field.top), (x, field.bottom))
        for y in range(field.top, field.bottom, self.scale):
            pygame.draw.line(self.surface, (26, 45, 61), (field.left, y), (field.right, y))
        pygame.draw.rect(self.surface, (72, 91, 109), field, 2)

        point = self.world_to_pixels
        self.surface.set_clip(field)
        for track_center, travel, _ in SERVICE_HALL.movers:
            center = np.asarray(track_center)
            displacement = np.asarray(travel)
            for i in range(9):
                offset = i / 4 - 1
                pygame.draw.circle(self.surface, (132, 83, 91),
                                   point(center + offset * displacement), 2)
        walls = np.asarray(state.static_obstacles)[len(SERVICE_HALL.bollards):]
        for wall in walls:
            pygame.draw.circle(self.surface, (85, 106, 123), point(wall), self.scale)
        for first, last in SERVICE_HALL.walls:
            pygame.draw.line(self.surface, (64, 79, 96), point(first), point(last),
                             2 * self.scale - 4)
            for end in (first, last):
                pygame.draw.circle(self.surface, (64, 79, 96), point(end), self.scale - 2)
        path = [point(p) for p in np.asarray(state.path_array)]
        for p in path:
            pygame.draw.circle(self.surface, (64, 109, 140), p, 2)
        position = np.asarray(state.agent_pos)
        center = point(position)
        if self.last_position is not None:
            self.distance += float(np.linalg.norm(position - self.last_position))
        self.last_position = position
        self.trail.append(center)
        if len(self.trail) > 1:
            pygame.draw.lines(self.surface, (38, 157, 184), False, self.trail, 3)
        for ray in np.asarray(obs.collision_rays):
            pygame.draw.line(self.surface, (48, 83, 106), center, point(position + ray))

        goal = point(np.asarray(state.goal_pos))
        pygame.draw.circle(self.surface, (38, 169, 111), goal, self.scale, 3)
        pygame.draw.circle(self.surface, (80, 220, 156), goal, 5)
        bollards = state.static_obstacles[:len(SERVICE_HALL.bollards)]
        for centers, fill, border in (
            (bollards, (65, 75, 94), (111, 127, 150)),
            (state.kinematic_obstacles, (151, 64, 72), (244, 115, 116)),
        ):
            for obstacle in np.asarray(centers):
                px = point(obstacle)
                pygame.draw.circle(self.surface, fill, px, round(self.scale * RADIUS))
                pygame.draw.circle(self.surface, border, px, round(self.scale * RADIUS), 2)

        pygame.draw.circle(self.surface, (61, 196, 222), center, round(self.scale * RADIUS))
        direction = np.asarray(state.agent_forward_dir)
        pygame.draw.line(self.surface, (15, 45, 60), center, point(position + RADIUS * direction), 5)
        self.surface.set_clip(None)
        y = field.bottom + 12
        telemetry = (f"{float(state.speed):.2f} units/s   |   "
                     f"{int(state.time) / params.fps:.1f} s   |   Return {reward:+.2f}"
                     f"   |   Driven {self.distance:.1f} units")
        self.label(telemetry, self.margin, y)
        self.label(status, field.right - 180, y, (99, 219, 173) if status == "Goal reached" else (239, 192, 113))
        self.label("Arrows / WASD to drive   |   R to reset   |   Esc to exit" if mode == "Manual"
                   else "Blue dots: planned route   |   Coral: moving obstacles   |   R to reset",
                   self.margin, y + 31, (148, 170, 192))

    def pixels(self):
        return pygame.surfarray.array3d(self.surface).transpose(1, 0, 2).copy()


def replay(args):
    key = jax.random.PRNGKey(args.seed)
    params, initial = RobotaxiEnv.init_params(key)
    model = load_model(args.checkpoint) if args.controller == "ppo" else None
    model_action = jax.jit(lambda obs: jnp.tanh(predict(model, obs.as_vector())[0]))
    mode = {"manual": "Manual", "route": "Route following", "ppo": "Trained PPO"}[args.controller]
    visible = not args.video
    frontend = RobotaxiFrontend(params, visible)
    clock = pygame.time.Clock()
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
    writer_context = imageio.get_writer(args.video, fps=params.fps, codec="libx264",
                                        pixelformat="yuv420p", macro_block_size=1) if args.video else nullcontext()
    try:
        with writer_context as writer:
            seed = args.seed
            obs, state = RobotaxiEnv.reset(key, params, initial, not args.nominal)
            total = 0.0
            status = "Driving"
            done = False
            end_frames = 0
            paused = False
            while True:
                if visible:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            return
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                            end_frames = params.fps * 2
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                            paused = not paused
                    if end_frames >= params.fps * 2:
                        if args.controller != "manual":
                            seed += 1
                        key = jax.random.PRNGKey(seed)
                        obs, state = RobotaxiEnv.reset(key, params, initial, not args.nominal)
                        done, end_frames, total, status = False, 0, 0.0, "Driving"
                        frontend.reset_trail()
                    if paused:
                        clock.tick(params.fps)
                        continue

                frontend.draw(state, obs, total, status, mode, seed, params)
                if writer:
                    writer.append_data(frontend.pixels())
                if visible:
                    pygame.display.flip()
                    clock.tick(params.fps)
                if done:
                    end_frames += 1
                    if writer and end_frames == params.fps:
                        break
                    if args.controller == "manual":
                        end_frames = min(end_frames, params.fps * 2 - 1)
                    continue

                if args.controller == "manual":
                    action = keyboard_action()
                elif args.controller == "route":
                    action = route_follower(obs, params)
                else:
                    action = model_action(obs)
                obs, state, reward, terminal, info = RobotaxiEnv.step(key, state, action, params)
                total += float(reward)
                done = bool(terminal)
                if done:
                    status = "Goal reached" if bool(info["success"]) else "Collision" if bool(info["collision"]) else "Time limit"
                    print(f"{mode} seed {seed}: {status}, {int(state.time)} steps, return {total:+.2f}", flush=True)
        if args.video:
            print(f"Saved {args.video}", flush=True)
    finally:
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("manual", "route", "ppo"), default="manual")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/room/policy.npz"))
    parser.add_argument("--seed", type=int, default=200_001)
    parser.add_argument("--nominal", action="store_true", help="Use the map's fixed initial conditions.")
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    if args.video and args.controller == "manual":
        parser.error("choose route or ppo to record a video")
    replay(args)


if __name__ == "__main__":
    main()
