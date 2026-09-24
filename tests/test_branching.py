"""The prefix tree a scenario sends forks exactly where its parameters say.

Each case pins the expected parent of every turn by hand. The conversation is
then driven through `run_conversation` against a model that records every
request, so the check covers the scenario's parent list, the runner's choice of
history, and the text itself: any two requests share exactly the history up to
their fork, then diverge inside the tag of the next turn.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings

from bench_cache.runner import run_conversation
from bench_cache.scenario import Scenario, load_scenario

# Small turns keep the recorded requests, and pytest's diffs of them on failure, fast.
TURN_TOKENS = 64

# (scenario, params, the parent of each turn in order; 0 is the bare system prompt)
CASES: dict[str, tuple[str, dict[str, Any], list[int]]] = {
    "linear": ("linear", {"n_turns": 4}, [0, 1, 2, 3]),
    # A -> B, then B -> C1..C4
    "width-4": ("branch", {"trunk": 2, "branches": 4}, [0, 1, 2, 2, 2, 2]),
    # A -> B, then two branches of 3 turns each
    "long": ("branch", {"trunk": 2, "branches": 2, "branch_turns": 3}, [0, 1, 2, 3, 4, 2, 6, 7]),
    # A -> B, B splits 3 ways, each of those splits 2 ways
    "nested": ("branch", {"trunk": 2, "branches": [3, 2]}, [0, 1, 2, 3, 3, 2, 6, 6, 2, 9, 9]),
    # as nested, but the first-level branches run 2 turns before splitting
    "nested-long": (
        "branch",
        {"trunk": 2, "branches": [3, 2], "branch_turns": [2, 1]},
        [0, 1, 2, 3, 4, 4, 2, 7, 8, 8, 2, 11, 12, 12],
    ),
    # A1..A5, then two branches from A3 and two from A5
    "branch-at": ("branch", {"trunk": 5, "branch_at": [3, 5], "branches": 2}, [0, 1, 2, 3, 4, 3, 3, 5, 5]),
    # no trunk: independent roots that share only the system prompt
    "no-trunk": ("branch", {"trunk": 0, "branches": 3}, [0, 0, 0]),
}


def _scenario(name: str, params: dict[str, Any]) -> Scenario:
    return load_scenario(name).with_params({"turn_tokens": TURN_TOKENS} | params)


def _depth(parents: list[int], turn: int) -> int:
    return 0 if turn == 0 else 1 + _depth(parents, parents[turn - 1])


def _path(parents: list[int], turn: int) -> list[int]:
    """The turns from the root down to `turn`, inclusive."""
    return [] if turn == 0 else [*_path(parents, parents[turn - 1]), turn]


def _fork(parents: list[int], a: int, b: int) -> int:
    """The deepest turn both `a` and `b` continue from (0 if only the system prompt is shared)."""
    common = [x for x, y in zip(_path(parents, a), _path(parents, b), strict=False) if x == y]
    return common[-1] if common else 0


@dataclass
class _Recorder:
    """A target whose model answers "OK" and records each request as the flat sequence
    of text blocks a provider would see: instructions, then prompts and replies."""

    name: str = "recorder"
    requests: list[list[str]] = field(default_factory=list)

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        blocks = [info.instructions or ""]
        for message in messages:
            for part in message.parts:
                if isinstance(message, ModelRequest) and isinstance(part, UserPromptPart):
                    assert isinstance(part.content, str)
                    blocks.append(part.content)
                elif isinstance(part, TextPart):
                    blocks.append(part.content)
        self.requests.append(blocks)
        return ModelResponse(parts=[TextPart("OK")])

    def build(self) -> tuple[Model, ModelSettings]:
        return FunctionModel(self._respond), {}


def _record(scenario: str, params: dict[str, Any]) -> list[list[str]]:
    recorder = _Recorder()
    asyncio.run(run_conversation(recorder, _scenario(scenario, params)))  # type: ignore[arg-type]
    return recorder.requests


def _shared(a: list[str], b: list[str]) -> int:
    """How many leading blocks two requests have in common."""
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))


@pytest.mark.parametrize("case", CASES)
def test_parents(case: str) -> None:
    scenario, params, expected = CASES[case]
    _, turns = _scenario(scenario, params).build("key")
    assert [i - 1 if t.parent is None else t.parent for i, t in enumerate(turns, start=1)] == expected


@pytest.mark.parametrize("case", CASES)
def test_requests_fork_where_expected(case: str) -> None:
    scenario, params, parents = CASES[case]
    requests = _record(scenario, params)
    assert len(requests) == len(parents)

    for a, b in itertools.combinations(range(1, len(parents) + 1), 2):
        ra, rb = requests[a - 1], requests[b - 1]
        fork = _fork(parents, a, b)
        if fork == a:
            # b continues a: a's whole request is a prefix of b's, so b can reuse a's cache.
            assert rb[: len(ra)] == ra, f"turn {b} does not extend turn {a}"
            continue
        # system prompt, then a prompt and a reply for every turn up to the fork
        expected = 1 + 2 * _depth(parents, fork)
        assert _shared(ra, rb) == expected, f"turns {a} and {b} should fork after turn {fork}"
        # The first blocks to differ are sibling turns at the same depth: the same
        # filler under a different tag, so the prefix breaks inside the tag line.
        tag_a, body_a = ra[expected].split("\n", 1)
        tag_b, body_b = rb[expected].split("\n", 1)
        assert tag_a != tag_b
        assert body_a == body_b, f"turns {a} and {b} differ past the tag at their fork"


def test_runs_share_no_prefix() -> None:
    """Two runs of the same scenario diverge at the session key, before any cacheable text."""
    params = CASES["width-4"][1]
    first, second = _record("branch", params)[0], _record("branch", params)[0]
    assert first[0].split("\n", 1)[0] != second[0].split("\n", 1)[0]
    assert first[0].split("\n", 1)[1] == second[0].split("\n", 1)[1]
