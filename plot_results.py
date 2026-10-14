"""
Turns results/*.csv from evaluate.py into report figures and a summary table.

  python plot_results.py                       # reads results/, writes results/*.png and results/summary.md
  python plot_results.py --dir results --out figures

Outputs
  fig_robustness.png : success rate, fall rate and wall contacts vs disturbance severity, per policy
  fig_replanning.png : success rate with and without online replanning when an unmapped obstacle blocks the route
  summary.md         : the same numbers as tables (paste into the report; also the accessible table view)
Error bars on rates are 95% Wilson intervals; on the wall-contact mean they are +/- 1 standard error.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical slots (blue, orange, aqua) on a light surface; text uses ink tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]                      # second channel besides colour
INK, INK2, GRID, SURFACE, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#9a9893"


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n (returns lo, hi as fractions)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def style(ax, title, ylabel, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def policy_order(df):
    names = sorted(df["policy"].unique(), key=lambda n: (n != "baseline", n))   # baseline first
    return names, {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}


def fig_robustness(df, path):
    names, color = policy_order(df)
    df = df.assign(success=df["success"].astype(bool), fell=df["fell"].astype(bool))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), facecolor=SURFACE)
    panels = [("success", "Success rate (%)", "Reached the goal"),
              ("fell", "Fall rate (%)", "Robot fell over"),
              ("wall_events", "Wall contacts per run", "Collisions with walls")]
    for ax, (col, ylabel, title) in zip(axes, panels):
        style(ax, title, ylabel, "Disturbance severity (1 = DR training range)")
        for i, name in enumerate(names):
            d = df[df["policy"] == name]
            xs, ys, lo, hi = [], [], [], []
            for sev, g in d.groupby("severity"):
                xs.append(sev)
                if col == "wall_events":
                    m, se = g[col].mean(), g[col].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0
                    ys.append(m); lo.append(m - se); hi.append(m + se)
                else:
                    k, n = int(g[col].sum()), len(g)
                    a, b = wilson(k, n)
                    ys.append(100 * k / n); lo.append(100 * a); hi.append(100 * b)
            ys, lo, hi = map(np.array, (ys, lo, hi))
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], color=color[name], lw=2, marker=MARKERS[i % 3],
                        ms=7, mfc=color[name], mec=SURFACE, mew=1.5, capsize=3, elinewidth=1,
                        label=name)
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        color=INK, fontsize=9, va="center")        # direct label in ink, not series colour
        if col != "wall_events":
            ax.set_ylim(-3, 103)
        ax.set_xlim(left=-0.1, right=df["severity"].max() * 1.22)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    n = df.groupby(["policy", "severity"]).size().min()
    fig.suptitle(f"Robustness on unseen random mazes  (n = {n} mazes per point, same mazes and "
                 f"disturbances for every policy)", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_replanning(df, path):
    names, _ = policy_order(df)
    df = df.assign(success=df["success"].astype(bool), replan=df["replan"].astype(str) == "True")
    fig, ax = plt.subplots(figsize=(1.6 + 2.2 * len(names), 4.0), facecolor=SURFACE)
    style(ax, "Unmapped obstacle on the route", "Success rate (%)")
    w = 0.34
    for j, (replan, label, col) in enumerate([(False, "No replanning", MUTED), (True, "With replanning", SERIES[0])]):
        for i, name in enumerate(names):
            g = df[(df["policy"] == name) & (df["replan"] == replan)]
            k, n = int(g["success"].sum()), len(g)
            a, b = wilson(k, n)
            x = i + (j - 0.5) * (w + 0.02)
            r = 100 * k / max(n, 1)
            ax.bar(x, r, w, color=col, label=label if i == 0 else None, zorder=2)
            ax.errorbar(x, r, yerr=[[r - 100 * a], [100 * b - r]], color=INK2, lw=1, capsize=3, zorder=3)
            ax.text(x, 100 * b + 3, f"{k}/{n}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def summary_tables(rob, rep):
    out = ["# Results summary\n"]

    def row(g):
        k, n = int(g["success"].sum()), len(g)
        lo, hi = wilson(k, n)
        ok = g[g["success"].astype(bool)]
        return dict(n=n, **{"success %": f"{100 * k / n:.0f} [{100 * lo:.0f}-{100 * hi:.0f}]",
                            "fall %": f"{100 * g['fell'].astype(bool).mean():.0f}",
                            "stuck %": f"{100 * (g['reason'] == 'stuck').mean():.0f}",
                            "wall contacts/run": f"{g['wall_events'].mean():.1f}",
                            "SPL (successes)": f"{ok['spl'].mean():.2f}" if len(ok) else "-",
                            "time s (successes)": f"{ok['time_s'].mean():.0f}" if len(ok) else "-"})

    if rob is not None:
        t = pd.DataFrame({k: row(g) for k, g in rob.groupby(["policy", "severity"])}).T
        t.index.names = ["policy", "severity"]
        out += ["## Experiment 1: robustness vs disturbance severity (success % with 95% Wilson interval)\n",
                t.reset_index().to_markdown(index=False), "\n"]
    if rep is not None:
        rep = rep.assign(replan=rep["replan"].astype(str) == "True")
        t = pd.DataFrame({k: dict(row(g), replans=f"{g['n_replans'].mean():.1f}")
                          for k, g in rep.groupby(["policy", "replan"])}).T
        t.index.names = ["policy", "replanning"]
        out += ["## Experiment 2: unmapped obstacle, with vs without replanning\n",
                t.reset_index().to_markdown(index=False), "\n"]
    return "\n".join(out)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.join(here, "results"), help="folder with robustness.csv / replan.csv")
    p.add_argument("--out", default=None, help="output folder (default: same as --dir)")
    a = p.parse_args()
    out = a.out or a.dir
    os.makedirs(out, exist_ok=True)

    def load(name):
        f = os.path.join(a.dir, name)
        if not os.path.exists(f):
            return None
        d = pd.read_csv(f)
        d = d[~d["reason"].astype(str).str.startswith("error")]          # drop crashed episodes
        return d if len(d) else None

    rob, rep = load("robustness.csv"), load("replan.csv")
    if rob is None and rep is None:
        raise SystemExit(f"no result files found in {a.dir} (run evaluate.py first)")
    if rob is not None:
        fig_robustness(rob, os.path.join(out, "fig_robustness.png"))
        print("wrote", os.path.join(out, "fig_robustness.png"))
    if rep is not None:
        fig_replanning(rep, os.path.join(out, "fig_replanning.png"))
        print("wrote", os.path.join(out, "fig_replanning.png"))
    md = summary_tables(rob, rep)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(md)
    print("wrote", os.path.join(out, "summary.md"))
    print("\n" + md)


if __name__ == "__main__":
    main()
