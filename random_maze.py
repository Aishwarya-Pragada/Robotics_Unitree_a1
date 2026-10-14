"""
Procedural maze generator for the unseen-maze evaluation.

Same grid convention as layouts.py / maze_generator.py:
    0 = free, 1 = wall, 2 = start (top-left), 3 = goal (bottom-right); each cell is 1 m x 1 m.

How it works
  1. Carve a "perfect" maze (exactly one route between any two cells) with a recursive
     backtracker on the even-coordinate cells.
  2. Knock out a few extra wall cells (loop_prob) so there are several routes. This matters for
     the replanning experiment: if an obstacle blocks the only corridor there is nothing to
     replan to, so loops guarantee that a detour exists.

A maze is fully determined by (rows, cols, seed, loop_prob), so results are reproducible.

  python random_maze.py --rows 7 --cols 9 --seed 3
"""
import argparse
import numpy as np

from astar import astar
from maze_generator import get_start_and_goal, print_maze


def generate_maze(rows=7, cols=9, seed=0, loop_prob=0.15):
    """rows and cols must be odd (>= 3). Returns an int8 array."""
    if rows % 2 == 0 or cols % 2 == 0 or rows < 3 or cols < 3:
        raise ValueError("rows and cols must be odd and >= 3")
    rng = np.random.default_rng(seed)
    maze = np.ones((rows, cols), dtype=np.int8)

    # recursive backtracker over the cells that sit on even rows and even columns
    maze[0, 0] = 0
    stack = [(0, 0)]
    while stack:
        r, c = stack[-1]
        options = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 1:
                options.append((nr, nc, dr, dc))
        if options:
            nr, nc, dr, dc = options[rng.integers(len(options))]
            maze[r + dr // 2, c + dc // 2] = 0      # open the wall between the two cells
            maze[nr, nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()

    # loops: wall cells that sit between two corridor cells (exactly one odd coordinate)
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1 and (r % 2) != (c % 2) and rng.random() < loop_prob:
                maze[r, c] = 0

    maze[0, 0] = 2
    maze[rows - 1, cols - 1] = 3
    return maze


def maze_stats(maze):
    """Shortest-path length (cells), number of turns on it, and how many free cells there are."""
    start, goal = get_start_and_goal(maze)
    path = astar(maze, start, goal)
    turns = 0
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            turns += 1
    return dict(path_cells=len(path), turns=turns, free_cells=int(np.sum(maze != 1)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=7)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loop-prob", type=float, default=0.15)
    a = p.parse_args()
    m = generate_maze(a.rows, a.cols, a.seed, a.loop_prob)
    start, goal = get_start_and_goal(m)
    path = astar(m, start, goal)
    print_maze(m, path)
    print(maze_stats(m))
