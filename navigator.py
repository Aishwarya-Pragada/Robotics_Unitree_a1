"""
Navigator: the high-level planner with online replanning.

It keeps its OWN belief map of the maze (starts as the known layout) and plans on it with A*.
The real world can differ from that map: an obstacle block may be sitting in a corridor that the
planner does not know about. The robot only "sees" an obstacle once it is within `sense_radius`
metres (a stand-in for a range sensor). When that happens, or when the robot has been pushed
off its route, the navigator replans with A* from the robot's current cell and swaps in a new
waypoint path for the pure-pursuit follower.

  nav = Navigator(maze, v_max=0.5, replan=True)
  vx, vy, wz = nav.update(x, y, yaw, hidden_obstacle_cells)    # call every control step
"""
import numpy as np

from astar import astar
from maze_generator import get_start_and_goal
from maze_scene import grid_to_world, world_to_grid, prune_collinear, CELL
from follower import WaypointFollower, dist_to_polyline


class Navigator:
    def __init__(self, maze, v_max=0.5, replan=True, sense_radius=1.5, off_path_tol=0.8):
        self.belief = np.array(maze, copy=True)          # what the planner believes
        self.rows, self.cols = self.belief.shape
        self.start, self.goal = get_start_and_goal(self.belief)
        self.v_max = v_max
        self.replan_enabled = replan
        self.sense_radius = sense_radius
        self.off_path_tol = off_path_tol
        self.n_replans = 0
        self.failed = False                              # True if no route exists any more
        self.done = False
        self._plan(self.start, robot_xy=None)

    # ------------------------------------------------------------------ planning
    def _cell_of(self, x, y):
        """Cell containing (x, y); if that cell is a wall (robot clipped a corner), nearest free one."""
        r, c = world_to_grid(x, y, self.rows, self.cols)
        if self.belief[r, c] != 1:
            return r, c
        best, best_d = (r, c), 1e9
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.belief[nr, nc] != 1:
                    wx, wy = grid_to_world(nr, nc, self.rows)
                    d = np.hypot(wx - x, wy - y)
                    if d < best_d:
                        best, best_d = (nr, nc), d
        return best

    def _plan(self, from_cell, robot_xy):
        """A* on the belief map -> waypoint corners. The first point is the robot's real position."""
        path = astar(self.belief, tuple(int(v) for v in from_cell), tuple(int(v) for v in self.goal))
        if path is None:
            self.failed = True
            return False
        pts = [np.array(grid_to_world(int(r), int(c), self.rows), dtype=float) for r, c in path]
        if robot_xy is not None and len(pts) > 1:
            pts[0] = np.array(robot_xy, dtype=float)     # start from where the robot really is
        if len(pts) == 1:                                # already in the goal cell
            pts = [pts[0], pts[0] + 1e-3]
        self.cells = path
        self.corners = prune_collinear(pts)
        self.follower = WaypointFollower(self.corners, v_max=self.v_max)
        self.plan_len = (len(path) - 1) * CELL
        return True

    # ------------------------------------------------------------------- sensing
    def _sense(self, x, y, hidden_cells):
        """Add every hidden obstacle within sense_radius to the belief map. Returns True if new."""
        new = False
        for (r, c) in hidden_cells:
            if self.belief[r, c] == 1:
                continue
            wx, wy = grid_to_world(r, c, self.rows)
            if np.hypot(wx - x, wy - y) <= self.sense_radius:
                self.belief[r, c] = 1
                new = True
        return new

    # ------------------------------------------------------------------- control
    def update(self, x, y, yaw, hidden_cells=()):
        """Returns the (vx, vy, yaw_rate) command for this control step."""
        if self.failed:
            return 0.0, 0.0, 0.0
        newly_seen = self._sense(x, y, hidden_cells)

        if self.replan_enabled and not self.done:
            on_path = any(self.belief[r, c] == 1 for r, c in self.cells)
            off_path = dist_to_polyline(np.array([x, y]), self.corners) > self.off_path_tol
            if (newly_seen and on_path) or off_path:
                if self._plan(self._cell_of(x, y), robot_xy=(x, y)):
                    self.n_replans += 1

        if self.failed:
            return 0.0, 0.0, 0.0
        vx, vy, wz = self.follower.command(x, y, yaw)
        self.done = self.follower.done
        return vx, vy, wz


def pick_obstacle_cell(maze, rng, min_gap=3):
    """Choose a cell on the shortest path to block, such that a detour still exists.

    Skips the first/last `min_gap` cells of the path. Returns (row, col) or None if every
    candidate would disconnect start from goal (e.g. a perfect maze with a single corridor).
    """
    start, goal = get_start_and_goal(maze)
    path = astar(maze, start, goal)
    candidates = [cell for cell in path[min_gap:len(path) - min_gap]]
    rng.shuffle(candidates)
    for r, c in candidates:
        blocked = np.array(maze, copy=True)
        blocked[r, c] = 1
        if astar(blocked, start, goal) is not None:
            return int(r), int(c)
    return None
