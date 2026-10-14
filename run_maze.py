"""
Full pipeline demo: A* planner (+ replanning) -> waypoint follower -> trot policy -> maze goal.

  python run_maze.py --layout layout1 --run runs/trot_v1 --video maze1.mp4
  python run_maze.py --layout layout2 --run runs/trot_v1                    # live viewer
  python run_maze.py --layout layout3 --run runs/trot_v1 --no-render

New for the final project:
  python run_maze.py --maze-seed 7 --run runs/dr_v1                         # unseen random maze
  python run_maze.py --maze-seed 7 --obstacles 1 --run runs/dr_v1           # hidden obstacle + replanning
  python run_maze.py --maze-seed 7 --obstacles 1 --no-replan --run runs/dr_v1   # ... without replanning
  python run_maze.py --maze-seed 7 --severity 1.5 --run runs/dr_v1          # pushes, low friction, ...
"""
import argparse
import os
import sys
import time
import numpy as np
import mujoco

from layouts import LAYOUTS
from maze_scene import HERE
from random_maze import generate_maze
from episode import run_episode, load_policy


class RenderHooks:
    """Live viewer or mp4 recording for episode.run_episode."""

    def __init__(self, video=None, cam="top", render=True):
        self.video, self.cam_mode, self.render = video, cam, render
        self.viewer = self.renderer = self.cam = None
        self.frames = []

    def setup(self, base, maze):
        rows, cols = maze.shape
        self.base = base
        if self.video:
            self.renderer = mujoco.Renderer(base.model, 480, 640)
        elif self.render:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(base.model, base.data)
        if not (self.renderer or self.viewer):
            return
        self.cam = self.viewer.cam if self.viewer else mujoco.MjvCamera()
        if self.cam_mode == "top":
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.lookat[:] = [(cols - 1) / 2, (rows - 1) / 2, 0.0]
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.25 * max(rows, cols) + 3, 90, -80
        else:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = base.trunk_id
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 135, -30

    def step(self, base):
        if self.viewer:
            if not self.viewer.is_running():
                return False
            self.viewer.sync()
            time.sleep(base.dt)
        if self.renderer:
            self.renderer.update_scene(base.data, camera=self.cam)
            self.frames.append(self.renderer.render())
        return True

    def close(self):
        if self.viewer:
            self.viewer.close()
        if self.renderer and self.frames:
            import imageio
            imageio.mimsave(self.video, self.frames, fps=int(round(1 / self.base.dt)))
            print("saved video:", self.video)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="layout1", choices=list(LAYOUTS))
    p.add_argument("--maze-seed", type=int, default=None, help="use a generated unseen maze instead of --layout")
    p.add_argument("--rows", type=int, default=7, help="generated maze rows (odd)")
    p.add_argument("--cols", type=int, default=9, help="generated maze cols (odd)")
    p.add_argument("--run", default=os.path.join(HERE, "runs", "trot_v1"))
    p.add_argument("--model", default=None)
    p.add_argument("--stats", default=None)
    p.add_argument("--speed", type=float, default=0.5, help="max forward command (m/s)")
    p.add_argument("--max-time", type=float, default=None, help="seconds (default: scales with path length)")
    p.add_argument("--obstacles", type=int, default=0, help="hidden obstacle blocks on the shortest path")
    p.add_argument("--no-replan", action="store_true", help="follow the first plan blindly")
    p.add_argument("--severity", type=float, default=0.0, help="disturbance severity (0 off, 1 = training range)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", default=None)
    p.add_argument("--cam", default="top", choices=["top", "track"])
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    if args.maze_seed is not None:
        maze = generate_maze(args.rows, args.cols, args.maze_seed)
        name = f"gen{args.maze_seed}"
    else:
        maze, name = LAYOUTS[args.layout], args.layout

    policy = load_policy(args.run, args.model, args.stats)
    hooks = None if (args.no_render and not args.video) else RenderHooks(args.video, args.cam, not args.no_render)
    res = run_episode(maze, policy, name=name, speed=args.speed, max_time=args.max_time,
                      replan=not args.no_replan, n_obstacles=args.obstacles, severity=args.severity,
                      seed=args.seed, draw_path=True, hooks=hooks)

    print(f"\n{name}: {maze.shape[0]}x{maze.shape[1]}, shortest route {res['opt_len']:.1f} m"
          f"{', ' + str(res['obstacles']) + ' hidden obstacle(s)' if res['obstacles'] else ''}")
    print("===== maze result =====")
    print(f"success               : {res['success']}  ({res['reason']})")
    print(f"fell                  : {res['fell']}")
    print(f"time                  : {res['time_s']:.1f} s")
    print(f"distance to goal      : {res['final_dist']:.2f} m")
    print(f"path length (optimal / actual): {res['opt_len']:.1f} m / {res['actual_len']:.1f} m   (SPL {res['spl']:.2f})")
    print(f"path deviation        : mean {res['mean_dev']:.2f} m, max {res['max_dev']:.2f} m")
    print(f"wall contacts         : {res['wall_events']} events ({res['wall_steps']} steps)")
    print(f"replans               : {res['n_replans']}   (obstacles discovered: {res['obstacles_seen']}/{res['obstacles']})")
    if args.severity > 0:
        print(f"sampled physics       : friction x{res['dr_friction']}, trunk mass x{res['dr_trunk_mass']}, motor gain x{res['dr_kp']}")
    sys.stdout.flush()
    os._exit(0)      # avoids the MuJoCo viewer segfault on exit


if __name__ == "__main__":
    main()
