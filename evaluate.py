"""
Experiment runner for the final project. Everything is headless, parallel and resumable.

  Experiment 1  "robustness": does domain randomization keep the robot upright and on route when
                conditions are worse than nominal? Every policy x severity x unseen random maze.
  Experiment 2  "replan":     does online replanning rescue the robot when an unmapped obstacle
                blocks its route? Every policy x {replan off, on} x unseen random maze.

  python evaluate.py --exp robustness --policies baseline=runs/trot_v1 dr=runs/dr_ft --n-mazes 20 --workers 4
  python evaluate.py --exp replan     --policies dr=runs/dr_ft --n-mazes 20 --workers 4
  python evaluate.py --exp all        --policies baseline=runs/trot_v1 dr=runs/dr_ft

Results are appended to results/<exp>.csv as each episode finishes. Re-running the same command
skips episodes that are already in the file (use --fresh to start over), so you can stop and
resume. Then: python plot_results.py

Design notes
  * The same maze seed and the same physics seed are used for every policy, so the comparison is
    paired: policies face identical mazes and identical sampled disturbances.
  * The RL policy is trained on flat ground only. It never sees any maze, so "unseen mazes" tests
    the whole pipeline, not memorization.
"""
import argparse
import csv
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [(5, 7), (7, 7), (7, 9)]                    # maze sizes cycled over seeds (rows, cols)
FIELDS = ["exp", "policy", "severity", "maze_seed", "rows", "cols", "replan", "n_obstacles",
          "success", "reason", "fell", "time_s", "final_dist", "first_plan_len", "opt_len",
          "actual_len", "spl", "mean_dev", "max_dev", "wall_events", "wall_steps", "n_replans",
          "obstacles_seen", "dr_friction", "dr_trunk_mass", "dr_kp"]


def key_of(row):
    return (row["exp"], row["policy"], float(row["severity"]), int(row["maze_seed"]),
            str(row["replan"]) == "True", int(row["n_obstacles"]))


# ------------------------------------------------------------------------------- worker side
def _init_worker():
    import torch
    torch.set_num_threads(1)                        # one core per worker, no oversubscription


def run_task(task):
    """Runs in a worker process. Never raises: failures come back as reason='error: ...'."""
    from random_maze import generate_maze
    from episode import run_episode, load_policy
    row = dict(exp=task["exp"], policy=task["policy"], severity=task["severity"],
               maze_seed=task["maze_seed"], rows=task["rows"], cols=task["cols"],
               replan=task["replan"], n_obstacles=task["n_obstacles"])
    try:
        maze = generate_maze(task["rows"], task["cols"], task["maze_seed"])
        policy = load_policy(task["run_dir"])
        res = run_episode(maze, policy, name=f"{task['exp']}_{task['maze_seed']}", speed=task["speed"],
                          replan=task["replan"], n_obstacles=task["n_obstacles"],
                          severity=task["severity"], seed=task["maze_seed"])
        row.update({k: res[k] for k in FIELDS if k in res})
    except Exception as e:                          # keep the sweep alive
        row.update(success=False, reason=f"error: {type(e).__name__}: {e}"[:120])
    return row


# ------------------------------------------------------------------------------ task building
def valid_obstacle_mazes(n, n_obstacles=1):
    """First n maze seeds (with their sizes) where an obstacle with a guaranteed detour fits."""
    from random_maze import generate_maze
    from episode import choose_obstacles
    out, seed = [], 0
    while len(out) < n and seed < 50 * n:
        rows, cols = SIZES[seed % len(SIZES)]
        maze = generate_maze(rows, cols, seed)
        if len(choose_obstacles(maze, n_obstacles, seed)) == n_obstacles:
            out.append((seed, rows, cols))
        seed += 1
    return out


def build_tasks(args, policies):
    tasks = []
    if args.exp in ("robustness", "all"):
        mazes = [(s, *SIZES[s % len(SIZES)]) for s in range(args.n_mazes)]
        for (name, run_dir), sev, (seed, r, c) in itertools.product(policies, args.severities, mazes):
            tasks.append(dict(exp="robustness", policy=name, run_dir=run_dir, severity=sev,
                              maze_seed=seed, rows=r, cols=c, replan=True, n_obstacles=0, speed=args.speed))
    if args.exp in ("replan", "all"):
        mazes = valid_obstacle_mazes(args.n_mazes)
        for (name, run_dir), replan, (seed, r, c) in itertools.product(policies, (False, True), mazes):
            tasks.append(dict(exp="replan", policy=name, run_dir=run_dir, severity=args.replan_severity,
                              maze_seed=seed, rows=r, cols=c, replan=replan, n_obstacles=1, speed=args.speed))
    return tasks


# ------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["robustness", "replan", "all"], default="all")
    p.add_argument("--policies", nargs="+", required=True, metavar="NAME=RUN_DIR",
                   help="e.g. baseline=runs/trot_v1 dr=runs/dr_ft")
    p.add_argument("--n-mazes", type=int, default=20, help="unseen mazes per condition")
    p.add_argument("--severities", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--replan-severity", type=float, default=1.0, help="disturbance level for experiment 2")
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--fresh", action="store_true", help="ignore and overwrite existing result files")
    args = p.parse_args()

    policies = []
    for spec in args.policies:
        name, _, run_dir = spec.partition("=")
        if not run_dir:
            sys.exit(f"--policies expects NAME=RUN_DIR, got '{spec}'")
        run_dir = os.path.abspath(run_dir)
        for f in ("model.zip", "vecnormalize.pkl"):
            if not os.path.exists(os.path.join(run_dir, f)):
                sys.exit(f"missing {f} in {run_dir}")
        policies.append((name, run_dir))

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = build_tasks(args, policies)

    # resume support: one CSV per experiment
    paths = {e: os.path.join(args.out_dir, f"{e}.csv") for e in ("robustness", "replan")}
    done = set()
    for e, path in paths.items():
        if args.fresh and os.path.exists(path):
            os.remove(path)
        if os.path.exists(path):
            with open(path, newline="") as f:
                done |= {key_of(r) for r in csv.DictReader(f)}
    todo = [t for t in tasks if key_of(t) not in done]
    print(f"{len(tasks)} episodes planned, {len(tasks) - len(todo)} already done, {len(todo)} to run "
          f"on {args.workers} worker(s)")
    if not todo:
        return

    def write(row):
        path = paths[row["exp"]]
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    t0, n_done = time.time(), 0

    def report(row):
        nonlocal n_done
        n_done += 1
        write(row)
        eta = (time.time() - t0) / n_done * (len(todo) - n_done)
        print(f"[{n_done:3d}/{len(todo)}] {row['exp']:10} {row['policy']:10} sev={float(row['severity']):.1f} "
              f"maze={int(row['maze_seed']):2d} replan={str(row['replan']):5} -> {row.get('reason', '?'):8} "
              f"(ETA {eta / 60:.1f} min)", flush=True)

    if args.workers <= 1:
        _init_worker()
        for t in todo:
            report(run_task(t))
    else:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker) as pool:
            for fut in as_completed([pool.submit(run_task, t) for t in todo]):
                report(fut.result())
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.out_dir}")


if __name__ == "__main__":
    main()
