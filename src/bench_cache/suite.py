"""Load suites: TOML files that pin down which conversations a run makes.

A suite is the reproducible half of a benchmark. It names each case and fixes
its scenario and parameters, so the same workload can be rerun later or against
other targets. Which targets to run and how many repeats are separate, chosen
per run on the command line (`-t`, `-T targets.toml`, `-n`), so one suite serves
a quick single-run check and a many-repeat measurement alike.

    description = "Linear and branching conversations"

    [[case]]
    name = "linear"
    scenario = "linear"
    params = { n_turns = 10, turn_tokens = 3000 }

    [[case]]
    name = "branch-t2-w4"        # branch at turn 2 into 4
    scenario = "branch"
    params = { trunk = 2, branches = 4 }

    [[case]]
    name = "width"
    scenario = "branch"
    params = { trunk = 2 }
    sweep = { branches = [2, 4, 8] }   # width.branches=2, width.branches=4, ...

Parameters a case leaves out keep the scenario's defaults. Each case's full
parameter set is recorded with its results, so later changes to a default can't
silently change what an old run meant.

A targets file lists target specs, one per string:

    targets = ["openai:gpt-5-mini", "openrouter:deepseek/deepseek-v4-flash@streamlake"]

Both kinds of file can also ship with the package, in `suites/` and
`target_lists/`, and are then referred to by bare name instead of a path.
"""

from __future__ import annotations

import itertools
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scenario import DEFAULT_PROMPTS_DIR, Scenario, load_scenario

BUILTIN_SUITES_DIR = Path(__file__).parent / "suites"
BUILTIN_TARGETS_DIR = Path(__file__).parent / "target_lists"

_CASE_KEYS = {"name", "scenario", "params", "sweep"}
_SAFE_NAME = re.compile(r"^[\w.=,+-]+$")
"""Case names become result file names."""


@dataclass(frozen=True)
class Case:
    name: str
    scenario: Scenario


@dataclass(frozen=True)
class Suite:
    name: str
    description: str
    cases: list[Case]
    source: Path | None = None
    """The TOML file the suite came from; None for a single scenario run from the CLI."""


def list_builtin(builtin_dir: Path) -> list[str]:
    return sorted(p.stem for p in builtin_dir.glob("*.toml"))


def resolve_file(ref: str, builtin_dir: Path, kind: str) -> Path:
    """A ref ending in `.toml` is a path; a bare name is a file shipped in `builtin_dir`."""
    if ref.endswith(".toml"):
        return Path(ref)
    path = builtin_dir / f"{ref}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no built-in {kind} {ref!r} (have: {list_builtin(builtin_dir)})")
    return path


def single(scenario: Scenario) -> Suite:
    """A one-case suite, for running a scenario by name."""
    return Suite(name=scenario.name, description=scenario.description, cases=[Case(scenario.name, scenario)])


def _fmt(v: Any) -> str:
    return ",".join(map(str, v)) if isinstance(v, list) else str(v)


def _expand(where: str, raw: dict[str, Any], prompts_dir: Path) -> list[Case]:
    if unknown := set(raw) - _CASE_KEYS:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; expected {sorted(_CASE_KEYS)}")
    if "scenario" not in raw:
        raise ValueError(f"{where}: missing `scenario`")
    scenario = load_scenario(raw["scenario"], prompts_dir)
    name = raw.get("name", scenario.name)
    params: dict[str, Any] = raw.get("params", {})
    sweep: dict[str, Any] = raw.get("sweep", {})
    if both := set(params) & set(sweep):
        raise ValueError(f"{where}: {sorted(both)} set in both `params` and `sweep`")
    if bad := [k for k, v in sweep.items() if not isinstance(v, list) or not v]:
        raise ValueError(f"{where}: sweep values must be non-empty lists, not {bad}")

    cases = []
    for combo in itertools.product(*sweep.values()):
        point = dict(zip(sweep, combo, strict=True))
        suffix = "".join(f".{k}={_fmt(v)}" for k, v in point.items())
        try:
            configured = scenario.with_params(params | point)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        cases.append(Case(name + suffix, configured))
    return cases


def load_suite(path: Path, prompts_dir: Path = DEFAULT_PROMPTS_DIR) -> Suite:
    with path.open("rb") as f:
        data = tomllib.load(f)
    if unknown := set(data) - {"description", "case"}:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; expected `description` and `[[case]]` tables")
    raw_cases = data.get("case", [])
    if not raw_cases:
        raise ValueError(f"{path}: no [[case]] tables")

    cases = [c for i, raw in enumerate(raw_cases, start=1) for c in _expand(f"{path} case {i}", raw, prompts_dir)]
    names = [c.name for c in cases]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"{path}: duplicate case name(s) {dupes}; give each [[case]] a distinct `name`")
    if unsafe := [n for n in names if not _SAFE_NAME.match(n)]:
        raise ValueError(f"{path}: case name(s) {unsafe} must be usable as file names (letters, digits, _.=,+-)")
    for c in cases:
        c.scenario.build("validate")  # catch a bad turn structure before any spend
    return Suite(name=path.stem, description=data.get("description", ""), cases=cases, source=path)


def load_targets_file(path: Path) -> list[str]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    targets = data.get("targets")
    if set(data) != {"targets"} or not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise ValueError(f'{path}: expected only `targets = ["provider:model_id", ...]`')
    return targets
