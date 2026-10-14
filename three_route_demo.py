"""
Three-route demo: the robot hits an obstacle, reroutes, hits a SECOND obstacle, reroutes again,
and finishes on the third route. All three routes are exactly the same length.

WHY THE ROUTES ARE EQUAL
  The maze is a "ladder": three long horizontal lanes (top, middle, bottom) joined by one free
  column on the far left and one on the far right. Start is the top-left corner, goal the
  bottom-right corner. Every route is "go right and go down", so no matter which lane you pick the
  length is the same:  (rows - 1) + (cols - 1) cells.  Lanes are 4 rows apart, so a sensor
  looking at one lane never sees an obstacle in another lane.

        S . . . . . . . . . . . . . . . .          lane 1 (top)
        . # # # # # # # # # # # # # # # .
        . # # # # # # # # # # # # # # # .
        . # # # # # # # # # # # # # # # .
        . . . . . . . . . . . . . . . . .          lane 2 (middle)
        . # # # # # # # # # # # # # # # .
        . # # # # # # # # # # # # # # # .
        . # # # # # # # # # # # # # # # .
        . . . . . . . . . . . . . . . . G          lane 3 (bottom)

WHAT HAPPENS
  1. A* plans the shortest route (all lanes tie; A* picks one).
  2. Obstacle 1 sits in the middle of that lane, UNMAPPED. The robot discovers it when it is close,
     turns back and replans onto a second lane.
  3. Obstacle 2 sits in the middle of that second lane. Same thing: discover, turn back, replan.
  4. The third lane is clear, and the robot walks it to the goal.

  The obstacle cells are not hard-coded: this script dry-runs the real Navigator along the robot's
  path to find exactly which lane A* will choose each time, then drops each obstacle on it.

USAGE (from the project folder)
  python three_route_demo.py --run runs/dr_ft                    # live viewer, top-down camera
  python three_route_demo.py --run runs/dr_ft --no-render        # numbers + saved picture only
  python three_route_demo.py --run runs/dr_ft --no-render --no-replan   # control: should get stuck
  python three_route_demo.py --run runs/dr_ft --severity 1.0     # add pushes / changed physics
  python three_route_demo.py --run runs/dr_ft --cam track        # camera follows the robot
"""
import argparse
import os
import sys
import numpy as np

import episode
import navigator as navigator_module
from astar import astar
from episode import run_episode, load_policy
from maze_generator import get_start_and_goal
from maze_scene import grid_to_world, HERE
from navigator import Navigator
from run_maze import RenderHooks


# --------------------------------------------------------------------------- the maze
def build_ladder_maze(cols=17, spacing=4, lanes=3):
    """Ladder maze: `lanes` horizontal lanes `spacing` rows apart, joined on the far left and far right.
    Returns (maze, lane_rows). 0 = free, 1 = wall, 2 = start (top-left), 3 = goal (bottom-right)."""
    rows = (lanes - 1) * spacing + 1
    maze = np.ones((rows, cols), dtype=np.int8)
    lane_rows = [i * spacing for i in range(lanes)]
    maze[lane_rows, :] = 0
    maze[:, 0] = 0
    maze[:, -1] = 0
    maze[0, 0] = 2
    maze[rows - 1, cols - 1] = 3
    return maze, lane_rows


def route_len(maze, start=None):
    s, g = get_start_and_goal(maze)
    path = astar(maze, tuple(start) if start else s, g)
    return None if path is None else len(path) - 1


def lane_of(path, col):
    """Row of the path cell that sits in column `col` (the lane the route uses there)."""
    hits = [r for r, c in path if c == col]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one route cell in column {col}, found {len(hits)}")
    return hits[0]


# ------------------------------------------- dry run: find where the robot will discover things
def walk_until_replan(nav, hidden, step=0.01):
    """Walk a point along nav's current route, calling the real navigator each step, until it
    replans. Returns the (x, y) where that happened, or None."""
    n0 = nav.n_replans
    pts = [np.array(grid_to_world(r, c, nav.rows), dtype=float) for r, c in nav.cells]
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        n = max(int(np.linalg.norm(d) / step), 1)
        for i in range(n + 1):
            p = a + d * i / n
            nav.update(p[0], p[1], 0.0, hidden)
            if nav.n_replans > n0:
                return tuple(p)
    return None


def plan_obstacles(maze, col1, col2, sense_radius):
    """Pick obstacle cells so the robot is forced through three different lanes, in order."""
    rows, cols = maze.shape
    nav = Navigator(maze, replan=True, sense_radius=sense_radius)
    path1 = list(nav.cells)
    r1 = lane_of(path1, col1)
    obs1 = (r1, col1)

    hidden = [obs1]
    at1 = walk_until_replan(nav, hidden)
    if at1 is None:
        raise RuntimeError("dry run: obstacle 1 never triggered a replan (is it on the route?)")
    path2 = list(nav.cells)                     # the route the robot takes after discovering obstacle 1
    r2 = lane_of(path2, col2)
    obs2 = (r2, col2)

    hidden = [obs1, obs2]
    at2 = walk_until_replan(nav, hidden)
    if at2 is None:
        raise RuntimeError("dry run: obstacle 2 never triggered a replan")
    path3 = list(nav.cells)                     # the route after discovering obstacle 2
    r3 = lane_of(path3, cols - 3) if path3[-1] == tuple(np.argwhere(maze == 3)[0]) else None
    return obs1, obs2, dict(lane1=r1, lane2=r2, lane3=r3, found1=at1, found2=at2, nav=nav)


# --------------------------------------------------------------------------- the picture
def save_picture(path, maze, obstacles, res, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("(matplotlib not installed, skipping picture)")
        return
    rows, cols = maze.shape
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for r in range(rows):
        for c in range(cols):
            x, y = grid_to_world(r, c, rows)
            if maze[r, c] == 1:
                ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc="#c9c8c3", ec="none"))
    for (r, c) in obstacles:
        x, y = grid_to_world(r, c, rows)
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc="#eb6834", ec="#0b0b0b", lw=1.2))
        ax.text(x, y, "X", ha="center", va="center", color="#0b0b0b", fontweight="bold")
    s, g = get_start_and_goal(maze)
    for (cell, col, label) in ((s, "#1baf7a", "Start"), (g, "#eda100", "Goal")):
        x, y = grid_to_world(cell[0], cell[1], rows)
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc=col, ec="none", alpha=.85))
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="#0b0b0b")
    t = res["trail"]
    ax.plot(t[:, 0], t[:, 1], color="#2a78d6", lw=2, label="robot's actual path")
    ax.set_xlim(-.7, cols - .3)
    ax.set_ylim(-.7, rows - .3)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("saved picture:", path)


# --------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=os.path.join(HERE, "runs", "trot_v1"))
    p.add_argument("--model", default=None)
    p.add_argument("--stats", default=None)
    p.add_argument("--cols", type=int, default=17, help="maze width in cells (longer = longer routes)")
    p.add_argument("--spacing", type=int, default=4, help="rows between lanes (keep >= 4)")
    p.add_argument("--obs1-col", type=int, default=8, help="column of obstacle 1 on the first route")
    p.add_argument("--obs2-col", type=int, default=11, help="column of obstacle 2 on the second route")
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--severity", type=float, default=0.0, help="disturbance severity (0 = none)")
    p.add_argument("--sense-radius", type=float, default=1.75,
                   help="how close the robot must be to notice an obstacle (m)")
    p.add_argument("--max-time", type=float, default=320.0, help="simulated seconds before giving up")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-replan", action="store_true", help="control run: follow the first plan blindly")
    p.add_argument("--no-render", action="store_true", help="no viewer window")
    p.add_argument("--cam", default="top", choices=["top", "track"])
    p.add_argument("--video", default=None, help="optional mp4 path")
    p.add_argument("--picture", default=os.path.join(HERE, "three_route_result.png"),
                   help="where to save the path picture ('' to skip)")
    a = p.parse_args()

    if a.spacing < 4:
        sys.exit("--spacing must be at least 4 so obstacles in one lane are never sensed from another")
    if not (2 <= a.obs1_col <= a.cols - 4 and 2 <= a.obs2_col <= a.cols - 4 and a.obs1_col != a.obs2_col):
        sys.exit("obstacle columns must be different and between 2 and cols-4")

    maze, lane_rows = build_ladder_maze(a.cols, a.spacing)
    rows, cols = maze.shape
    obs1, obs2, info = plan_obstacles(maze, a.obs1_col, a.obs2_col, a.sense_radius)

    # ---- prove the three routes are the same length
    m1 = maze.copy(); m1[obs1] = 1
    m12 = m1.copy(); m12[obs2] = 1
    L1, L2, L3 = route_len(maze), route_len(m1), route_len(m12)
    name = {lane_rows[0]: "top", lane_rows[1]: "middle", lane_rows[2]: "bottom"}
    print(f"maze: {rows} x {cols} cells ({rows - 1 + cols - 1} m from start to goal along any lane)")
    print(f"route 1 : {name[info['lane1']]:6} lane, {L1} m   <- first plan")
    print(f"route 2 : {name[info['lane2']]:6} lane, {L2} m   <- after obstacle 1 at cell {obs1}")
    final_lane = [r for r in lane_rows if r not in (info['lane1'], info['lane2'])][0]
    print(f"route 3 : {name[final_lane]:6} lane, {L3} m   <- after obstacle 2 at cell {obs2}")
    if not (L1 == L2 == L3):
        sys.exit(f"routes are not equal ({L1}, {L2}, {L3}): this should not happen")
    print("all three routes are the same length: OK\n")

    # ---- print each replan as it happens
    orig_plan = navigator_module.Navigator._plan
    count = {"n": 0}

    def traced(self, from_cell, robot_xy):
        ok = orig_plan(self, from_cell, robot_xy)
        if robot_xy is not None and ok:
            count["n"] += 1
            print(f"  >> replan #{count['n']}: robot at ({robot_xy[0]:.1f}, {robot_xy[1]:.1f}) m, "
                  f"new route is {self.plan_len:.0f} m from its position")
        return ok
    navigator_module.Navigator._plan = traced

    # run_episode normally picks random obstacle cells; hand it ours instead
    episode.choose_obstacles = lambda maze_, n, seed: [obs1, obs2]

    policy = load_policy(a.run, a.model, a.stats)
    hooks = None if (a.no_render and not a.video) else RenderHooks(a.video, a.cam, not a.no_render)
    print("running (a full run takes a few minutes of simulated time)...")
    res = run_episode(maze, policy, name="three_route", speed=a.speed, max_time=a.max_time,
                      replan=not a.no_replan, n_obstacles=2, severity=a.severity, seed=a.seed,
                      sense_radius=a.sense_radius, draw_path=True, hooks=hooks)

    # ---- results
    print("\n===== three-route result =====")
    print(f"success               : {res['success']}  ({res['reason']})")
    print(f"fell                  : {res['fell']}")
    print(f"replans               : {res['n_replans']}   (expected 2 with replanning on)")
    print(f"obstacles discovered  : {res['obstacles_seen']}/2")
    print(f"time                  : {res['time_s']:.1f} s")
    print(f"distance to goal      : {res['final_dist']:.2f} m")
    print(f"route length          : {L1} m each; distance actually walked {res['actual_len']:.1f} m "
          f"(the extra is the walking back after each obstacle)")
    print(f"wall contacts         : {res['wall_events']} events ({res['wall_steps']} steps)")
    if a.severity > 0:
        print(f"sampled physics       : friction x{res['dr_friction']}, trunk mass x{res['dr_trunk_mass']}, "
              f"motor gain x{res['dr_kp']}")
    if a.picture:
        mode = "no replanning" if a.no_replan else "with replanning"
        save_picture(a.picture, maze, [obs1, obs2], res,
                     f"Three equal routes, two unmapped obstacles ({mode}): "
                     f"{'reached the goal' if res['success'] else res['reason']}")
    sys.stdout.flush()
    os._exit(0)      # avoids the MuJoCo viewer segfault on exit


if __name__ == "__main__":
    main()