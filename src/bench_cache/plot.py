"""Render a results JSONL file as PNG charts next to it.

`<stem>_cached.png`: cached tokens per turn, one panel per target, one line per
conversation (trajectory) with the runs side by side in each turn's column, hit/miss
markers, and a mark on each first turn showing whether it read another run's cache.
A branching scenario is drawn as a tree: branches at the same depth share a column.
`<stem>_hits.png`: the same hits as a grid, one row per target and one box per
turn, each box filled to the share of conversations that hit on that turn.

A hit means the cache grew relative to the parent (usually previous) turn (see `cache_grew`).
Hits are recomputed from the cached-token sequence, so older result files plot
with the current definition too.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

from .runner import cache_grew

# Validated categorical order (light surface); runs beyond 8 fold into neutral gray.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
WARM = "#e34948"
"""A first turn that read cache: it can only have come from another run."""


def _lane_color(lane: int) -> str:
    return SERIES[lane % (len(SERIES) - 1)]  # the last series colour (red) is reserved for WARM


def _tokens(v: float, _pos: object = None) -> str:
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


Row = dict[str, Any]
"""One turn as written to the results JSONL."""

Runs = dict[str, dict[str, list[Row]]]
"""target -> run_key -> turns (sorted), in the order the runs happened."""


def _load(jsonl: Path) -> tuple[Row, Runs]:
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{jsonl} has no rows")
    runs: Runs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        runs[r["target"]][r["run_key"]].append(r)
    for trajectories in runs.values():
        for turns in trajectories.values():
            turns.sort(key=lambda t: t["turn"])
    return rows[0], runs


def _parent(t: Row) -> int:
    """The turn this one continued from; results written before branching existed have no `parent`."""
    return int(t.get("parent", t["turn"] - 1))


def _tree_layout(parents: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Place each turn at (depth, lane). A turn's first child continues its lane;
    each later child (a branch) starts a new lane below."""
    pos: dict[int, tuple[int, int]] = {0: (0, 0)}  # 0 is the bare system prompt
    continued: set[int] = set()
    n_lanes = 1
    for turn in sorted(parents):
        depth, lane = pos.get(parents[turn], (0, 0))
        if parents[turn] in continued:
            lane, n_lanes = n_lanes, n_lanes + 1
        continued.add(parents[turn])
        pos[turn] = (depth + 1, lane)
    del pos[0]
    return pos


def _run_slots(pos: dict[int, tuple[int, int]], n_runs: int) -> dict[tuple[int, int], float]:
    """x for each (turn, run index): the turn's depth column, split into one group per
    branch at that depth, with the runs side by side in the group (run 1 leftmost)."""
    lanes_at: dict[int, list[int]] = defaultdict(list)
    for depth, lane in sorted(pos.values()):
        lanes_at[depth].append(lane)
    xs: dict[tuple[int, int], float] = {}
    for turn, (depth, lane) in pos.items():
        group_w = 0.8 / len(lanes_at[depth])
        center = depth - 0.4 + group_w * (lanes_at[depth].index(lane) + 0.5)
        step = min(group_w * 0.8 / n_runs, 0.07)  # tight enough to read as one cluster per branch
        for i in range(n_runs):
            xs[turn, i] = center + (i - (n_runs - 1) / 2) * step
    return xs


def _header(fig: Figure, title: str, first: Row) -> None:
    height = fig.get_figheight()
    params = " ".join(f"{k}={v}" for k, v in (first.get("params") or {}).items())
    name = first["scenario"]
    if first.get("case", name) != name:  # a suite case: show which scenario it runs
        name = f"{first['case']} ({name})"
    fig.text(0.01, 1 - 0.15 / height, f"{title}: {name}", fontweight="bold", va="top")
    fig.text(0.01, 1 - 0.42 / height, params, color=TEXT_2, fontsize=9, va="top")


def render(jsonl: Path) -> list[Path]:
    """Write every chart for a results file; returns the PNG paths."""
    stem = jsonl.with_suffix("")
    return [
        plot_cached(jsonl, stem.with_name(f"{stem.name}_cached.png")),
        plot_hits(jsonl, stem.with_name(f"{stem.name}_hits.png")),
    ]


def plot_cached(jsonl: Path, out: Path) -> Path:
    first, runs = _load(jsonl)
    n = len(runs)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "text.color": TEXT})
    width = max(6.4 * ncols, 8.5)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, 4.2 * nrows + 1.4), sharey=True, squeeze=False, facecolor=SURFACE
    )
    layouts: dict[str, tuple[dict[int, tuple[int, int]], dict[int, int]]] = {}

    for ax, (target, trajectories) in zip(axes.flat, runs.items(), strict=False):
        ax.set_facecolor(SURFACE)
        parents = {t["turn"]: _parent(t) for turns in trajectories.values() for t in turns}
        pos = _tree_layout(parents)
        layouts[target] = pos, parents
        xs = _run_slots(pos, len(trajectories))
        prefix: dict[int, list[int]] = defaultdict(list)
        hits = turns_judged = starts = cold = cached_total = cacheable_total = 0

        for i, turns in enumerate(trajectories.values()):
            cached = {t["turn"]: t["cache_read_tokens"] for t in turns}
            for t in turns:
                turn, y, p = t["turn"], t["cache_read_tokens"], _parent(t)
                x, color = xs[turn, i], _lane_color(pos[turn][1])
                prefix[turn].append(t["prev_input_tokens"])
                cached_total += y
                cacheable_total += t["prev_input_tokens"]
                if p not in cached:
                    # The conversation's first turn: its key is new, so any cache read came from another run.
                    starts += 1
                    cold += y == 0
                    marker = {"marker": "o", "mfc": SURFACE, "mec": color} if y == 0 else {"marker": "D", "color": WARM}
                    ax.plot(x, y, ms=6, mew=1.5, zorder=4, clip_on=False, **marker)
                    continue
                # Each turn joins its parent, so a branch fans out from the fork.
                ax.plot([xs[p, i], x], [cached[p], y], color=color, lw=1.5, zorder=3, solid_capstyle="round")
                hit = cache_grew(y, cached[p])
                hits += hit
                turns_judged += 1
                ax.plot(
                    x,
                    y,
                    marker="o" if hit else "X",
                    ms=6 if hit else 9,
                    color=color,
                    mec=SURFACE,
                    mew=1.5,
                    zorder=4,
                    clip_on=False,  # misses at 0 sit on the baseline
                )

        # The most each turn could reuse, drawn along the same tree through each branch's centre.
        centre = {turn: statistics.mean(xs[turn, i] for i in range(len(trajectories))) for turn in pos}
        ceiling = {turn: statistics.median(v) for turn, v in prefix.items()}
        for turn, p in parents.items():
            if p in pos:
                ax.plot(
                    [centre[p], centre[turn]],
                    [ceiling[p], ceiling[turn]],
                    color=MUTED,
                    lw=1.5,
                    ls=(0, (4, 3)),
                    zorder=2,
                )

        reuse = cached_total / cacheable_total if cacheable_total else 0.0
        ax.set_title(target, loc="left", fontweight="bold", color=TEXT, pad=22)
        ax.text(
            0,
            1.03,
            f"{cold}/{starts} runs started cold · {hits}/{turns_judged} turns grew the cache · "
            f"{reuse:.1%} of reusable prefix reused",
            transform=ax.transAxes,
            color=TEXT_2,
            fontsize=9,
            va="bottom",
        )
        depth = max(d for d, _ in pos.values())
        ax.set_xlim(0.5, depth + 0.5)
        ax.set_xticks(range(1, depth + 1))
        ax.yaxis.set_major_formatter(FuncFormatter(_tokens))
        ax.set_xlabel("turn", color=TEXT_2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=TEXT_2, length=0)
        ax.set_ylim(bottom=0)

    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("cached tokens", color=TEXT_2)

    # One colour per branch, labelled from the first target (every target runs the same scenario).
    pos, parents = next(iter(layouts.values()))
    n_lanes = 1 + max(lane for _, lane in pos.values())
    main_label = "runs side by side, 1st leftmost"
    if n_lanes > 1:
        main_label = f"branch 1 ({main_label})"
    handles = [Line2D([], [], color=_lane_color(0), lw=2, label=main_label)]
    for lane in range(1, n_lanes):
        first_turn = min(turn for turn, (_, turn_lane) in pos.items() if turn_lane == lane)
        fork = parents[first_turn]
        if fork not in pos:
            origin = "the system prompt"
        elif n_lanes > 2:  # say which branch it forks from; with nested branches several share a turn
            origin = f"branch {pos[fork][1] + 1} turn {pos[fork][0]}"
        else:
            origin = f"turn {pos[fork][0]}"
        handles.append(Line2D([], [], color=_lane_color(lane), lw=2, label=f"branch {lane + 1} (from {origin})"))
    handles += [
        Line2D([], [], color=TEXT_2, marker="o", ls="", ms=6, label="hit: cached grew"),
        Line2D([], [], color=TEXT_2, marker="X", ls="", ms=9, label="miss: cached same or less"),
        Line2D([], [], color=TEXT_2, marker="o", mfc=SURFACE, mew=1.5, ls="", ms=6, label="first turn cold (no cache)"),
        Line2D([], [], color=WARM, marker="D", ls="", ms=6, label="first turn read another run's cache"),
        Line2D([], [], color=MUTED, lw=1.5, ls=(0, (4, 3)), label="reusable prefix (parent turn input)"),
    ]

    height = fig.get_figheight()
    _header(fig, "Cached tokens per turn", first)
    legend_cols = min(len(handles), 4 if ncols > 1 else 3)
    legend_rows = math.ceil(len(handles) / legend_cols)
    fig.legend(handles=handles, loc="lower center", ncol=legend_cols, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, (0.1 + 0.25 * legend_rows) / height, 1, 1 - 0.65 / height))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


HIT = SERIES[0]
MISS = GRID


def plot_hits(jsonl: Path, out: Path) -> Path:
    """One row per target (more when the conversation branches), one box per turn;
    each box is filled to the share of runs that hit."""
    first, runs = _load(jsonl)
    layouts = {
        target: _tree_layout({t["turn"]: _parent(t) for turns in trajectories.values() for t in turns})
        for target, trajectories in runs.items()
    }
    n_turns = max(depth for pos in layouts.values() for depth, _ in pos.values())
    lanes = {target: 1 + max(lane for _, lane in pos.values()) for target, pos in layouts.items()}
    pitch = 1.45  # row spacing in cell units: box + per-turn label below it
    gap = 0.4 if max(lanes.values()) > 1 else 0.0  # extra space between targets, so branch rows stay grouped

    cell = 0.42  # inches per cell
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "text.color": TEXT})
    fig, ax = plt.subplots(facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    y0 = 0.0
    target_ys = []
    for target, trajectories in runs.items():
        target_ys.append(y0)
        pos = layouts[target]
        hits_per_turn: dict[int, int] = defaultdict(int)
        judged_per_turn: dict[int, int] = defaultdict(int)
        parents: dict[int, int] = {}
        for turns in trajectories.values():
            cached = {t["turn"]: t["cache_read_tokens"] for t in turns}
            for t in turns:
                parents[t["turn"]] = _parent(t)
                if (p := _parent(t)) in cached:
                    hits_per_turn[t["turn"]] += cache_grew(t["cache_read_tokens"], cached[p])
                    judged_per_turn[t["turn"]] += 1

        for turn, (x, lane) in pos.items():
            y = y0 + lane * pitch
            parent_x, parent_lane = pos.get(parents[turn], (0, lane))
            if parent_lane != lane:
                # Branch: elbow from the parent's right edge, down the gap between columns, into this box.
                px, py = parent_x, y0 + parent_lane * pitch
                ax.plot([px + 0.44, px + 0.5, px + 0.5, x - 0.44], [py, py, y, y], color=MUTED, lw=1.2, zorder=1)
            left, bottom = x - 0.44, y + 0.42  # y axis is inverted: bottom edge is y + 0.42
            judged = judged_per_turn[turn]
            if not judged:
                # turn 1 has no previous turn to compare against
                ax.add_patch(Rectangle((left, y - 0.42), 0.88, 0.84, fill=False, ec=GRID, lw=1))
                continue
            share = hits_per_turn[turn] / judged
            ax.add_patch(Rectangle((left, y - 0.42), 0.88, 0.84, fc=MISS, ec="none"))
            ax.add_patch(Rectangle((left, bottom), 0.88, -0.84 * share, fc=HIT, ec="none"))
            ax.text(x, y + 0.5, f"{hits_per_turn[turn]}/{judged}", ha="center", va="top", color=TEXT_2, fontsize=8)

        total_hits, total_judged = sum(hits_per_turn.values()), sum(judged_per_turn.values())
        ax.text(n_turns + 0.7, y0, f"{total_hits}/{total_judged}", va="center", color=TEXT_2, fontsize=9)
        y0 += lanes[target] * pitch + gap

    last_y = y0 - pitch - gap  # the bottom row's centre
    ax.set_xlim(0.5, n_turns + 0.5)
    ax.set_ylim(last_y + 0.9, -0.5)
    ax.set_yticks(target_ys, list(runs), fontweight="bold")
    ax.set_xticks(range(1, n_turns + 1))
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelcolor=TEXT_2)
    ax.tick_params(axis="y", labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Lay out in inches so a box is `cell` square whatever the target names' length.
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    label_w = max(t.get_window_extent(renderer).width for t in ax.get_yticklabels()) / fig.dpi + 0.25
    axes_w, axes_h = cell * n_turns, cell * (last_y + 1.4)
    top, bottom, right = 1.05, 0.6, 0.9  # header + turn numbers; legend; per-target totals
    width = max(label_w + axes_w + right, 8.5)
    height = top + axes_h + bottom
    fig.set_size_inches(width, height)
    fig.subplots_adjust(
        left=label_w / width, right=(label_w + axes_w) / width, top=1 - top / height, bottom=bottom / height
    )
    ax.annotate(
        "turn",
        (1, 1),
        xycoords="axes fraction",
        xytext=(6, 4),
        textcoords="offset points",
        color=TEXT_2,
        fontsize=9,
        va="bottom",
    )

    n_runs = sorted({len(t) for t in runs.values()})
    runs_label = f"{n_runs[0]}" if len(n_runs) == 1 else f"{n_runs[0]}-{n_runs[-1]}"
    handles = [
        Patch(fc=HIT, label="runs where the cache grew (hit)"),
        Patch(fc=MISS, label="runs where it stayed or dropped (miss)"),
        Patch(fill=False, ec=GRID, label="turn 1 (nothing to compare)"),
    ]
    _header(fig, f"Cache hit rate per turn across {runs_label} runs", first)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out
