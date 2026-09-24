"""bench-cache CLI: run prompt scenarios against targets and report cache behaviour."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai.exceptions import ModelHTTPError

from .plot import render
from .runner import ConversationResult, run_conversation
from .scenario import DEFAULT_PROMPTS_DIR, list_scenarios, load_scenario
from .targets import TARGETS, list_hosts, resolve_target

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _print_result(r: ConversationResult) -> None:
    params = " ".join(f"{k}={v}" for k, v in r.params.items())
    print(f"\n{r.target}  scenario={r.scenario}  repeat={r.repeat}  key={r.run_key}  {params}".rstrip())
    header = f"{'turn':>4} {'expect':>6} {'input':>7} {'cached':>7} {'write':>6} {'output':>6} {'hit%':>6} {'reuse%':>7} {'lat s':>6} {'cost $':>10} {'upstream':<12} verdict"
    print(header)
    print("-" * len(header))
    for t in r.turns:
        reuse = f"{t.prefix_reuse:7.1%}" if t.prefix_reuse is not None else f"{'-':>7}"
        cost = t.provider_details.get("cost")
        cost_s = f"{cost:10.6f}" if isinstance(cost, int | float) else f"{'-':>10}"
        upstream = str(t.provider_details.get("downstream_provider", "-"))
        print(
            f"{t.turn:>4} {t.expect:>6} {t.input_tokens:>7} {t.cache_read_tokens:>7} {t.cache_write_tokens:>6} "
            f"{t.output_tokens:>6} {t.hit_rate:6.1%} {reuse} {t.latency_s:6.2f} {cost_s} {upstream:<12} {t.verdict}"
        )
    print("-" * len(header))
    print(_summary(r.input_tokens, r.cache_read_tokens, r.cacheable_tokens))


def _summary(input_tokens: int, cached: int, cacheable: int) -> str:
    hit = cached / input_tokens if input_tokens else 0.0
    reuse = cached / cacheable if cacheable else 0.0
    return (
        f"total input {input_tokens}, cached {cached} "
        f"= {hit:.1%} effective hit rate, {reuse:.1%} of the {cacheable} reusable prefix tokens"
    )


def _write_jsonl(results: list[ConversationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in results:
            for t in r.turns:
                row = {
                    "scenario": r.scenario,
                    "target": r.target,
                    "run_key": r.run_key,
                    "repeat": r.repeat,
                    "params": r.params,
                }
                f.write(json.dumps(row | t.to_json(), default=str) + "\n")


async def _run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario, args.prompts_dir).with_params(dict(args.param))
    try:
        targets = [resolve_target(spec) for spec in args.target]
    except (KeyError, ValueError) as e:
        print(f"{e.args[0]}; see `bench-cache list`", file=sys.stderr)
        return 2

    results: list[ConversationResult] = []
    errors = 0
    for target in targets:
        for rep in range(args.repeats):
            try:
                r = await run_conversation(
                    target,
                    scenario,
                    repeat=rep,
                    turn_delay_s=args.turn_delay,
                )
            except ModelHTTPError as e:
                print(f"\n{target.name}  repeat={rep}: HTTP {e.status_code}: {e.body}", file=sys.stderr)
                errors += 1
                continue
            _print_result(r)
            results.append(r)

    if len(results) > 1:
        print("\nper target, all repeats:")
        for target_name in args.target:
            rs = [r for r in results if r.target == target_name]
            if rs:
                inp, cached, cacheable = (
                    sum(getattr(r, a) for r in rs) for a in ("input_tokens", "cache_read_tokens", "cacheable_tokens")
                )
                print(f"  {target_name} (n={len(rs)}): {_summary(inp, cached, cacheable)}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out_dir / f"{stamp}_{scenario.name}.jsonl"
    failed = sum(not r.passed for r in results)
    print(f"\n{len(results) - failed}/{len(results)} conversations met expectations, {errors} errored.")
    if results:
        _write_jsonl(results, out)
        print("Wrote", *[out, *render(out)], sep="\n  ")
    return 1 if failed or errors else 0


def _list(args: argparse.Namespace) -> int:
    print("scenarios:")
    for name in list_scenarios(args.prompts_dir):
        scenario = load_scenario(name, args.prompts_dir)
        print(f"  {name:<28} {scenario.description}")
        if scenario.params:
            print(f"  {'':<28} params: " + " ".join(f"{k}={v}" for k, v in scenario.params.items()))
    print("targets:")
    for t in TARGETS.values():
        print(f"  {t.name:<40} {t.description}")
    print("\npin upstream hosts with name@host[,host...]; `bench-cache hosts <target>` lists host slugs")
    return 0


def _hosts(args: argparse.Namespace) -> int:
    target = resolve_target(args.target)
    print(f"{target.model_id} upstream hosts (USD per 1M tokens):")
    print(f"  {'slug':<16} {'provider':<16} {'tag':<22} {'input':>8} {'cache read':>10}")
    for h in list_hosts(target.model_id):
        price = lambda v: f"{float(v) * 1e6:.4f}" if v is not None else "-"  # noqa: E731
        print(f"  {h['slug']:<16} {h['name']:<16} {h['tag']:<22} {price(h['prompt']):>8} {price(h['cache_read']):>10}")
    return 0


def _scalar(raw: str) -> int | float | str:
    for conv in (int, float):
        try:
            return conv(raw)
        except ValueError:
            pass
    return raw


def _param(text: str) -> tuple[str, int | float | str | list[int | float | str]]:
    """Parse `name=value`; numbers become int/float, and a comma-separated value becomes a list."""
    name, sep, raw = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected name=value, got {text!r}")
    if "," in raw:
        return name, [_scalar(v) for v in raw.split(",")]
    return name, _scalar(raw)


def main() -> None:
    p = argparse.ArgumentParser(prog="bench-cache", description=__doc__)
    p.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list scenarios and targets").set_defaults(func=_list)

    h = sub.add_parser("hosts", help="list OpenRouter upstream hosts that serve a target's model")
    h.add_argument("target")
    h.set_defaults(func=_hosts)

    r = sub.add_parser("run", help="run a scenario against one or more targets")
    r.add_argument("scenario")
    r.add_argument("-t", "--target", action="append", required=True, help="target name, optionally name@host[,host...] to pin upstream hosts (repeatable)")
    r.add_argument(
        "-p", "--param", type=_param, action="append", default=[], metavar="NAME=VALUE", help="scenario parameter"
    )
    r.add_argument("-n", "--repeats", type=int, default=1)
    r.add_argument("--turn-delay", type=float, default=0.0, help="seconds to wait between turns")
    r.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    r.set_defaults(func=lambda a: asyncio.run(_run(a)))

    args = p.parse_args()
    sys.exit(args.func(args))
