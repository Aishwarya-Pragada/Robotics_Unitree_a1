# Robust Hierarchical Navigation for a Quadruped (Unitree A1) in Unseen Mazes

A Unitree A1 in MuJoCo that navigates mazes it has never seen, while being pushed around and
walking on changing physics. Three layers, each doing the job it is best at:

```
 maze grid ──► A* planner ──► pure-pursuit follower ──► RL trot policy (PPO) ──► MuJoCo A1
              (+ online          (path -> vx, vy, yaw     (velocity-conditioned,
               replanning)        velocity command)        domain-randomized)
```

**Why this design makes sense for RL.** The RL policy is trained on *flat ground only*, with random
velocity commands. It never sees a maze, so it cannot memorize one. What it learns is the hard part:
staying balanced and tracking a velocity command under uncertainty. Route finding is solved by A*,
which is exact for a known grid. Evaluation then uses mazes the system has never seen.

## Contributions (the "innovation" for the report)

1. **Domain randomization + pushes + sensor noise** for the locomotion policy (`rl_env.py`).
   Per episode: friction, trunk mass, motor strength; random velocity pushes; observation noise.
   A single `severity` knob (0 = nominal, 1 = training range, >1 = harder than trained) lets us
   test *out-of-distribution* robustness.
2. **Online replanning with limited sensing** (`navigator.py`). Obstacles the planner does not know
   about are discovered within 1.5 m, and A* replans from the robot's current cell. It also replans
   if a push throws the robot off its route.
3. **A reproducible evaluation harness** (`evaluate.py`, `random_maze.py`): generated unseen mazes
   with guaranteed solutions and detours, paired comparisons, 95% confidence intervals.
4. *(optional)* **Curriculum**: `--dr-ramp-steps` raises disturbance severity gradually during training.

## Files

| File | Status | Purpose |
|---|---|---|
| `rl_env.py` | modified | Env. Added domain randomization, pushes, noise (all OFF by default) |
| `train.py` | modified | Flags `--randomize`, `--dr-scale`, `--dr-ramp-steps`, `--init-from` (warm start) |
| `maze_env.py` | modified | Movable obstacle blocks (`place_obstacle`) |
| `maze_scene.py` | modified | `n_obstacles` option, `world_to_grid` |
| `follower.py` | modified | Guards against zero-length path segments |
| `run_maze.py` | modified | Visual demo, now supports generated mazes, obstacles, replanning, severity |
| `random_maze.py` | **new** | Procedural maze generator (solvable, with loops) |
| `navigator.py` | **new** | Belief map, sensing, replanning |
| `episode.py` | **new** | One full navigation episode + metrics (shared by demo and evaluation) |
| `evaluate.py` | **new** | Parallel, resumable experiments, writes `results/*.csv` |
| `plot_results.py` | **new** | Figures and summary tables for the report |

Unchanged: `a1.xml`, `scene.xml`, `astar.py`, `layouts.py`, `maze_generator.py`, `test.py`,
`check_env.py`, `stand.py`.

## Setup (Ubuntu, CPU only)

```bash
pip install -r requirements.txt
python check_env.py                     # robot should stand with zero action
```

## Workflow

```bash
# 1. Train the randomized policy. You already have the baseline in runs/trot_v1 (5M steps).
#    CPU-friendly option: fine-tune it with randomization on (about 2M steps):
python train.py --run-name dr_ft  --randomize --init-from runs/trot_v1 --timesteps 2000000
#    Fair control: give the baseline the SAME extra steps, without randomization:
python train.py --run-name base_ft --init-from runs/trot_v1 --timesteps 2000000
#    (From scratch instead: python train.py --run-name dr_v1 --randomize --dr-ramp-steps 1500000)
#    Watch it: tensorboard --logdir runs

# 2. Run the experiments (about 3 min per 100 episodes per 2 cores; resumable)
python evaluate.py --exp all --policies baseline=runs/base_ft dr=runs/dr_ft --n-mazes 20

# 3. Figures + tables
python plot_results.py                  # -> results/fig_robustness.png, fig_replanning.png, summary.md

# 4. Videos for the presentation
python run_maze.py --maze-seed 7 --run runs/dr_ft --severity 1.5 --video robust.mp4
python run_maze.py --maze-seed 7 --obstacles 1 --run runs/dr_ft --video replan.mp4
python run_maze.py --maze-seed 7 --obstacles 1 --no-replan --run runs/dr_ft --video no_replan.mp4
```

Your original scripts still work: `python run_maze.py --layout layout1 --run runs/trot_v1`.

## Experiments

**Experiment 1, robustness.** Every policy x severity {0, 0.5, 1, 1.5, 2} x 20 unseen random mazes.
Same mazes and same sampled disturbances for every policy. Metrics: success rate, fall rate,
wall contacts, SPL (path efficiency), time. Hypothesis: the randomized policy degrades more
slowly beyond severity 1.

**Experiment 2, replanning.** A hidden block sits on the shortest route (a detour always exists).
Compare replanning off vs on. Hypothesis: without replanning the robot gets stuck; with it, it detours.

Preliminary measurement (baseline policy `trot_v1` only, 20 mazes per point, run on a 2-core
machine): success 100% at severity 0, 0.5 and 1.0; 75% at 1.5; 35% at 2.0 (65% falls). Replanning
experiment on 8 mazes: 0/8 without, 8/8 with replanning. The randomized policy's numbers come from
your own training run.

## Suggested report outline

1. Problem and motivation (why RL for locomotion, why not RL for the maze)
2. System architecture (the diagram above)
3. Method: reward, observation, domain randomization table (ranges in `rl_env.py: DR_DEFAULT`),
   replanning logic
4. Experimental setup (mazes, metrics, paired design, 95% CIs)
5. Results (the two figures + `summary.md`)
6. Discussion and limitations
7. Future work

## Limitations (state these honestly in the report)

* Simulation only; no sim-to-real transfer was attempted.
* "Sensing" is an idealized radius check, not a real range sensor or camera, and ignores line of sight.
* Obstacles are full-cell blocks and appear only on the shortest route. Dynamic (moving) obstacles
  are not modeled.
* A* works on a 4-connected 1 m grid; the maze has no uneven terrain.
* Disturbances are velocity pushes, not forces, and the ranges are hand-chosen.

## Future work

Learned perception (depth or lidar rays) instead of a radius check; terrain curriculum and rough
floors; a memory-based policy that explores without a map; moving obstacles; hardware transfer.
