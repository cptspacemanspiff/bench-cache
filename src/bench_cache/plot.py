"""Render a results JSONL file as PNG charts next to it.

`<stem>_cached.png`: cached tokens per turn, one panel per target, one line per
conversation (trajectory), with hit/miss markers.
`<stem>_hits.png`: the same hits as a binary grid, one row per conversation and
one column per turn.

A hit means the cache grew relative to the previous turn (see `cache_grew`).
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
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .runner import cache_grew

# Validated categorical order (light surface); runs beyond 8 fold into neutral gray.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


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


def _header(fig: Figure, title: str, first: Row) -> None:
    height = fig.get_figheight()
    params = " ".join(f"{k}={v}" for k, v in (first.get("params") or {}).items())
    fig.text(0.01, 1 - 0.15 / height, f"{title}: {first['scenario']}", fontweight="bold", va="top")
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
    max_runs = 0

    for ax, (target, trajectories) in zip(axes.flat, runs.items(), strict=False):
        ax.set_facecolor(SURFACE)
        by_turn: dict[int, list[int]] = defaultdict(list)
        hits = turns_judged = cached_total = cacheable_total = 0

        for i, turns in enumerate(trajectories.values()):
            xs = [t["turn"] for t in turns]
            ys = [t["cache_read_tokens"] for t in turns]
            color = SERIES[i] if i < len(SERIES) else MUTED
            ax.plot(xs, ys, color=color, lw=2, zorder=3, solid_capstyle="round")
            for j, (x, y) in enumerate(zip(xs, ys, strict=True)):
                if j == 0:
                    continue  # turn 1 has no previous turn to compare against
                hit = cache_grew(y, ys[j - 1])
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
            for t in turns:
                by_turn[t["turn"]].append(t["prev_input_tokens"])
                cached_total += t["cache_read_tokens"]
                cacheable_total += t["prev_input_tokens"]
            max_runs = max(max_runs, len(trajectories))

        ceiling_x = sorted(by_turn)
        ax.plot(
            ceiling_x,
            [statistics.median(by_turn[x]) for x in ceiling_x],
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
            f"{hits}/{turns_judged} turns grew the cache · {reuse:.1%} of reusable prefix reused",
            transform=ax.transAxes,
            color=TEXT_2,
            fontsize=9,
            va="bottom",
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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

    handles = [
        Line2D([], [], color=SERIES[i] if i < len(SERIES) else MUTED, lw=2, label=f"run {i + 1}")
        for i in range(min(max_runs, len(SERIES)))
    ]
    if max_runs > len(SERIES):
        handles.append(Line2D([], [], color=MUTED, lw=2, label=f"runs {len(SERIES) + 1}-{max_runs}"))
    handles += [
        Line2D([], [], color=TEXT_2, marker="o", ls="", ms=6, label="hit: cached grew"),
        Line2D([], [], color=TEXT_2, marker="X", ls="", ms=9, label="miss: cached same or less"),
        Line2D([], [], color=MUTED, lw=1.5, ls=(0, (4, 3)), label="reusable prefix (prev. turn input)"),
    ]

    height = fig.get_figheight()
    _header(fig, "Cached tokens per turn", first)
    legend_cols = min(len(handles), 6 if ncols > 1 else 3)
    legend_rows = math.ceil(len(handles) / legend_cols)
    fig.legend(handles=handles, loc="lower center", ncol=legend_cols, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, (0.1 + 0.25 * legend_rows) / height, 1, 1 - 0.65 / height))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


HIT = SERIES[0]
MISS = GRID


def plot_hits(jsonl: Path, out: Path) -> Path:
    """Binary grid: one row per conversation, one column per turn; filled = cache grew."""
    first, runs = _load(jsonl)
    n_turns = max(len(turns) for trajectories in runs.values() for turns in trajectories.values())
    n_rows = sum(len(trajectories) for trajectories in runs.values())

    cell = 0.42  # inches per cell
    panel_pad = 0.95  # title + subtitle + turn labels above each panel
    width = max(1.6 + cell * n_turns + 0.9, 8.5)
    height = 0.75 + len(runs) * panel_pad + cell * n_rows + 0.3 * len(runs) + 0.6
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "text.color": TEXT})
    fig, axes = plt.subplots(
        len(runs),
        1,
        figsize=(width, height),
        squeeze=False,
        facecolor=SURFACE,
        gridspec_kw={"height_ratios": [len(t) + 0.8 for t in runs.values()]},
    )

    for ax, (target, trajectories) in zip(axes[:, 0], runs.items(), strict=True):
        ax.set_facecolor(SURFACE)
        n = len(trajectories)
        hits_per_turn = [0] * (n_turns + 1)
        judged_per_turn = [0] * (n_turns + 1)
        total_hits = total_judged = 0

        for row, turns in enumerate(trajectories.values()):
            cached = [t["cache_read_tokens"] for t in turns]
            row_hits = 0
            for j, t in enumerate(turns):
                x, y = t["turn"], row
                corner = (x - 0.44, y - 0.42)
                if j == 0:
                    # turn 1 has no previous turn: outline only
                    ax.add_patch(Rectangle(corner, 0.88, 0.84, fill=False, ec=GRID, lw=1))
                    continue
                hit = cache_grew(cached[j], cached[j - 1])
                row_hits += hit
                hits_per_turn[x] += hit
                judged_per_turn[x] += 1
                ax.add_patch(Rectangle(corner, 0.88, 0.84, fc=HIT if hit else MISS, ec="none"))
                if not hit:
                    ax.plot(x, y, marker="X", ms=7, color=TEXT_2, mec=MISS, mew=1)
            total_hits += row_hits
            total_judged += len(turns) - 1
            ax.text(n_turns + 0.7, row, f"{row_hits}/{len(turns) - 1}", va="center", color=TEXT_2, fontsize=9)

        if n > 1:
            for x in range(2, n_turns + 1):
                if judged_per_turn[x]:
                    ax.text(
                        x,
                        n - 0.2,
                        f"{hits_per_turn[x]}/{judged_per_turn[x]}",
                        ha="center",
                        va="top",
                        color=TEXT_2,
                        fontsize=8,
                    )

        ax.set_xlim(0.5, n_turns + 0.5)
        ax.set_ylim(n - 0.5 + (0.6 if n > 1 else 0), -0.5)
        ax.set_aspect("equal")
        ax.set_yticks(range(n), [f"run {i + 1}" for i in range(n)])
        ax.set_xticks(range(1, n_turns + 1))
        ax.xaxis.tick_top()
        ax.tick_params(colors=TEXT_2, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Stack above the top tick labels (turn numbers, ~18pt tall): subtitle, then title.
        ax.set_title(target, loc="left", fontweight="bold", color=TEXT, pad=36)
        ax.annotate(
            f"{total_hits}/{total_judged} turns grew the cache",
            (0, 1),
            xycoords="axes fraction",
            xytext=(0, 20),
            textcoords="offset points",
            color=TEXT_2,
            fontsize=9,
            va="bottom",
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

    handles = [
        Patch(fc=HIT, label="cache grew (hit)"),
        Line2D(
            [],
            [],
            color=TEXT_2,
            marker="X",
            ls="",
            ms=7,
            mfc=TEXT_2,
            markeredgecolor=MISS,
            label="cache same or dropped (miss)",
        ),
        Patch(fill=False, ec=GRID, label="turn 1 (nothing to compare)"),
    ]
    _header(fig, "Cache hit per turn", first)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.45 / height, 1, 1 - 0.65 / height))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out
