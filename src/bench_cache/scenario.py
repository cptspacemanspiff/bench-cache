"""Load prompt-sequence scenarios: built-in modules in this package, or plain
Python files in the prompts folder.

A scenario module defines two functions, both taking the run key first:

    system(key, ...) -> str
    turns(key, ...) -> list[Turn]

Any further keyword arguments with defaults become scenario parameters, which can
be overridden from the CLI (`-p n_turns=10 -p turn_tokens=3000`). Each function
only receives the parameters it declares.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Expectation = Literal["miss", "hit", "any"]
PromptFn = Callable[..., Any]

DEFAULT_PROMPTS_DIR = Path("prompts")


@dataclass(frozen=True)
class Turn:
    prompt: str
    expect: Expectation = "any"


def _param_defaults(fn: PromptFn) -> dict[str, Any]:
    params = list(inspect.signature(fn).parameters.values())[1:]  # skip `key`
    return {p.name: p.default for p in params if p.default is not inspect.Parameter.empty}


def _call(fn: PromptFn, key: str, params: dict[str, Any]) -> Any:
    accepted = inspect.signature(fn).parameters
    return fn(key, **{k: v for k, v in params.items() if k in accepted})


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    system: PromptFn
    turns: PromptFn
    params: dict[str, Any] = field(default_factory=dict)

    def build(self, key: str) -> tuple[str, list[Turn]]:
        """Materialize the system prompt and turn list for one conversation."""
        system = _call(self.system, key, self.params)
        turns: list[Turn] = _call(self.turns, key, self.params)
        if not turns:
            raise ValueError(f"scenario {self.name!r} produced no turns with {self.params}")
        if bad := {t.expect for t in turns} - {"miss", "hit", "any"}:
            raise ValueError(f"scenario {self.name!r}: unknown expectations {bad}")
        return system, turns

    def with_params(self, overrides: dict[str, Any]) -> Scenario:
        if unknown := set(overrides) - set(self.params):
            raise ValueError(f"scenario {self.name!r} has no parameter(s) {sorted(unknown)}; has {sorted(self.params)}")
        return dataclasses.replace(self, params=self.params | overrides)


# Scenarios generated in code ship with the package; hand-written prompt
# sequences live as files in the prompts folder (which wins on a name clash).
BUILTIN_SCENARIOS = {"ok_filler": "bench_cache.ok_filler"}


def list_scenarios(prompts_dir: Path = DEFAULT_PROMPTS_DIR) -> list[str]:
    files = {p.stem for p in prompts_dir.glob("*.py") if not p.stem.startswith("_")}
    return sorted(files | set(BUILTIN_SCENARIOS))


def load_scenario(name: str, prompts_dir: Path = DEFAULT_PROMPTS_DIR) -> Scenario:
    path = prompts_dir / f"{name}.py"
    if path.is_file():
        spec = importlib.util.spec_from_file_location(f"bench_cache_prompts.{name}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        origin = str(path)
    elif name in BUILTIN_SCENARIOS:
        module = importlib.import_module(BUILTIN_SCENARIOS[name])
        origin = BUILTIN_SCENARIOS[name]
    else:
        raise FileNotFoundError(f"no scenario {name!r} (have: {list_scenarios(prompts_dir)})")

    for attr in ("system", "turns"):
        if not callable(getattr(module, attr, None)):
            raise ValueError(f"{origin}: scenario must define a `{attr}(key, ...)` function")

    params: dict[str, Any] = {}
    for fn in (module.system, module.turns):
        for k, v in _param_defaults(fn).items():
            if k in params and params[k] != v:
                raise ValueError(f"{origin}: parameter {k!r} has conflicting defaults {params[k]!r} and {v!r}")
            params[k] = v

    return Scenario(
        name=name,
        description=(module.__doc__ or "").strip().splitlines()[0] if module.__doc__ else "",
        system=module.system,
        turns=module.turns,
        params=params,
    )
