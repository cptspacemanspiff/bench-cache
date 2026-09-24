"""Drive a multi-turn conversation against a target and record per-turn cache usage."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse

from .scenario import Expectation, Scenario
from .targets import Target


def cache_grew(cached: int, prev_cached: int) -> bool:
    """A turn is a cache hit when it read more from cache than the previous turn did.

    The history only grows, so a working prefix cache reads strictly more each turn.
    Reading the same amount or less (e.g. 500 -> 1000 -> 500) means part of the
    cached prefix was lost, and counts as a miss.
    """
    return cached > prev_cached


def _cost_usd(response: ModelResponse) -> float | None:
    reported = (response.provider_details or {}).get("cost")  # OpenRouter reports the billed cost
    if isinstance(reported, int | float):
        return float(reported)
    try:
        return float(response.cost().total_price)
    except (LookupError, AssertionError):  # model unknown to genai-prices, or no model name
        return None


def _upstream(response: ModelResponse) -> str | None:
    host = (response.provider_details or {}).get("downstream_provider")  # OpenRouter
    return str(host) if host else None


@dataclass
class TurnStats:
    turn: int
    expect: Expectation
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    prev_input_tokens: int
    """Input tokens of the previous turn: the prefix this turn could have reused."""
    prev_cache_read_tokens: int
    latency_s: float
    verdict: str = ""
    cost_usd: float | None = None
    """Billed cost when the provider reports it, else estimated from genai-prices, else None."""
    upstream: str | None = None
    """The host that served the request, for routers that report it (OpenRouter)."""
    model_name: str | None = None
    response_id: str | None = None
    provider_details: dict[str, Any] = field(default_factory=dict)
    usage_details: dict[str, int] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.cache_read_tokens / self.input_tokens if self.input_tokens else 0.0

    @property
    def prefix_reuse(self) -> float | None:
        return self.cache_read_tokens / self.prev_input_tokens if self.prev_input_tokens else None

    def judge(self) -> str:
        match self.expect:
            case "any":
                return "-"
            case "miss":
                return "PASS" if self.cache_read_tokens == 0 else "FAIL"
            case "hit":
                return "PASS" if cache_grew(self.cache_read_tokens, self.prev_cache_read_tokens) else "FAIL"

    def to_json(self) -> dict[str, Any]:
        return asdict(self) | {"hit_rate": self.hit_rate, "prefix_reuse": self.prefix_reuse}


@dataclass
class ConversationResult:
    scenario: str
    target: str
    run_key: str
    repeat: int
    params: dict[str, Any]
    turns: list[TurnStats]

    @property
    def passed(self) -> bool:
        return all(t.verdict != "FAIL" for t in self.turns)

    @property
    def input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def cache_read_tokens(self) -> int:
        return sum(t.cache_read_tokens for t in self.turns)

    @property
    def cacheable_tokens(self) -> int:
        """Upper bound on cache reads: each turn could reuse at most the previous turn's input."""
        return sum(t.prev_input_tokens for t in self.turns)


async def run_conversation(
    target: Target,
    scenario: Scenario,
    *,
    repeat: int = 0,
    turn_delay_s: float = 0.0,
) -> ConversationResult:
    run_key = str(uuid.uuid4())
    system, scenario_turns = scenario.build(run_key)
    model, settings = target.build()
    agent = Agent(model, instructions=system, model_settings=settings)

    history: list[ModelMessage] = []
    turns: list[TurnStats] = []
    prev_input = prev_cached = 0
    for i, turn in enumerate(scenario_turns, start=1):
        if i > 1 and turn_delay_s:
            await asyncio.sleep(turn_delay_s)

        t0 = time.perf_counter()
        result = await agent.run(turn.prompt, message_history=history)
        latency = time.perf_counter() - t0
        history = result.all_messages()

        response = result.response
        u = response.usage
        stats = TurnStats(
            turn=i,
            expect=turn.expect,
            input_tokens=u.input_tokens,
            cache_read_tokens=u.cache_read_tokens,
            cache_write_tokens=u.cache_write_tokens,
            output_tokens=u.output_tokens,
            prev_input_tokens=prev_input,
            prev_cache_read_tokens=prev_cached,
            latency_s=latency,
            cost_usd=_cost_usd(response),
            upstream=_upstream(response),
            model_name=response.model_name,
            response_id=response.provider_response_id,
            provider_details=response.provider_details or {},
            usage_details=dict(u.details),
        )
        stats.verdict = stats.judge()
        turns.append(stats)
        prev_input, prev_cached = u.input_tokens, u.cache_read_tokens

    return ConversationResult(
        scenario=scenario.name,
        target=target.name,
        run_key=run_key,
        repeat=repeat,
        params=scenario.params,
        turns=turns,
    )
