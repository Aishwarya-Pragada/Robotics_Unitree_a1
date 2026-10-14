"""
One navigation episode, end to end:

    maze grid -> A* (+ online replanning) -> pure-pursuit follower -> RL trot policy -> MuJoCo

Used by run_maze.py (single run with viewer / video) and evaluate.py (many headless runs), so
both measure exactly the same thing. This module knows nothing about rendering: pass `hooks`
(an object with setup(env), step(env) -> bool, close()) if you want to draw the run.
"""
import os
import uuid
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from astar import astar
from maze_generator import get_start_and_goal
from maze_scene import build_scene, grid_to_world, solve, prune_collinear, HERE, CELL
from maze_env import MazeEnv
from navigator import Navigator, pick_obstacle_cell
from follower import dist_to_polyline

_POLICY_CACHE = {}


def load_policy(run_dir=None, model=None, stats=None):
    """Returns (PPO policy, path to its VecNormalize stats). Cached per process."""
    model = model or os.path.join(run_dir, "model.zip")
    stats = stats or os.path.join(run_dir, "vecnormalize.pkl")
    if (model, stats) not in _POLICY_CACHE:
        _POLICY_CACHE[(model, stats)] = PPO.load(model, device="cpu")
    return _POLICY_CACHE[(model, stats)], stats


def choose_obstacles(maze, n, seed):
    """Cells to block (each on the current shortest path, detour guaranteed). May return fewer than n."""
    rng = np.random.default_rng(seed + 777)
    work = np.array(maze, copy=True)
    cells = []
    for _ in range(n):
        cell = pick_obstacle_cell(work, rng)
        if cell is None:
            break
        work[cell] = 1
        cells.append(cell)
    return cells


def run_episode(maze, policy, *, name="run", speed=0.5, max_time=None, replan=True,
                n_obstacles=0, severity=0.0, dr_cfg=None, seed=0, sense_radius=1.5,
                draw_path=False, hooks=None, stall_s=15.0):
    """Run one episode and return a dict of metrics.

    severity   : domain-randomization severity (0 = nominal physics, 1 = training range, >1 harder)
    n_obstacles: unmapped blocks dropped on the shortest path; the planner only learns about
                 them when the robot gets within sense_radius metres
    replan     : whether the navigator may replan (False = follow the first plan blindly)
    """
    policy, stats = policy
    rows, cols = maze.shape
    start, goal = get_start_and_goal(maze)

    # ---- plan on the believed map, and work out what the oracle route would be on the true map
    hidden = choose_obstacles(maze, n_obstacles, seed) if n_obstacles else []
    true_maze = np.array(maze, copy=True)
    for cell in hidden:
        true_maze[cell] = 1
    opt_path = astar(true_maze, start, goal)
    opt_len = (len(opt_path) - 1) * CELL
    nav = Navigator(maze, v_max=speed, replan=replan, sense_radius=sense_radius)
    first_corners = nav.corners.copy()
    first_len = nav.plan_len
    if max_time is None:
        max_time = 30.0 + 3.0 * opt_len / speed

    d0 = first_corners[1] - first_corners[0]
    start_pose = (first_corners[0][0], first_corners[0][1], float(np.arctan2(d0[1], d0[0])))
    path_pts = solve(maze)[1] if draw_path else None

    # ---- build the scene, load it, delete the temporary XML again
    xml = build_scene(maze, f"tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}", path_pts=path_pts,
                      n_obstacles=len(hidden))
    try:
        venv = DummyVecEnv([lambda: MazeEnv(xml, start_pose=start_pose, episode_len_s=max_time + 5.0,
                                            randomize=severity > 0, dr_scale=severity, dr_cfg=dr_cfg)])
    finally:
        os.remove(xml)
    venv = VecNormalize.load(stats, venv)
    venv.training, venv.norm_reward = False, False
    base = venv.envs[0]
    dt = base.dt

    venv.seed(seed)
    obs = venv.reset()
    for slot, (r, c) in enumerate(hidden):
        base.place_obstacle(slot, *grid_to_world(r, c, rows))
    if hooks:
        hooks.setup(base, maze)

    # ---- run
    trail, dev = [], []
    wall_steps = wall_events = 0
    prev_wall = False
    reason = "timeout"
    hold = 0
    stall_n = int(stall_s / dt)
    for i in range(int(max_time / dt)):
        x, y, yaw = base.pose()
        vx, vy, wz = nav.update(x, y, yaw, hidden)
        if nav.failed:
            reason = "no_path"
            break
        base.set_command(vx, vy, wz)
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, _, infos = venv.step(action)
        info = infos[0]

        trail.append((x, y))
        dev.append(dist_to_polyline(np.array([x, y]), nav.corners))
        wall_steps += int(info["wall_hit"])
        wall_events += int(info["wall_hit"] and not prev_wall)
        prev_wall = info["wall_hit"]

        if hooks and hooks.step(base) is False:
            reason = "aborted"
            break
        if info["fell"]:
            reason = "fell"
            break
        if nav.done:
            hold += 1
            if hold > int(1.5 / dt):                    # stand still ~1.5 s so the stop is visible
                reason = "success"
                break
        elif len(trail) > stall_n and len(trail) % 25 == 0:
            # stuck = the robot stayed inside a ~25 cm box for stall_s seconds. (Net displacement
            # would wrongly flag a legitimate out-and-back detour after a replan.)
            if np.ptp(np.array(trail[-stall_n:]), axis=0).max() < 0.25:
                reason = "stuck"
                break
    if hooks:
        hooks.close()
    params = dict(base.dr_params)
    venv.close()

    # ---- metrics
    trail = np.array(trail) if trail else np.zeros((1, 2))
    travelled = float(np.sum(np.linalg.norm(np.diff(trail, axis=0), axis=1))) if len(trail) > 1 else 0.0
    gx, gy = grid_to_world(goal[0], goal[1], rows)
    success = reason == "success"
    return dict(
        success=success,
        reason=reason,
        fell=reason == "fell",
        time_s=round(len(trail) * dt, 2),
        final_dist=round(float(np.hypot(trail[-1][0] - gx, trail[-1][1] - gy)), 3),
        first_plan_len=round(first_len, 2),
        opt_len=round(opt_len, 2),
        actual_len=round(travelled, 2),
        spl=round(opt_len / max(travelled, opt_len), 3) if success else 0.0,
        mean_dev=round(float(np.mean(dev)), 3) if dev else 0.0,
        max_dev=round(float(np.max(dev)), 3) if dev else 0.0,
        wall_events=wall_events,
        wall_steps=wall_steps,
        n_replans=nav.n_replans,
        obstacles=len(hidden),
        obstacles_seen=int(np.sum(nav.belief != maze)),
        dr_friction=round(params["friction"], 3),
        dr_trunk_mass=round(params["trunk_mass"], 3),
        dr_kp=round(params["kp"], 3),
        corners=nav.corners,                           # final planned path (for plotting / debugging)
        trail=trail,
    )
