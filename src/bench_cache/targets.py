"""Registry of provider/model targets to benchmark.

Each target builds a pydantic-ai model plus the settings sent with every request.
Add new providers here; the runner only depends on `Target.build()`.

A target spec may pin OpenRouter upstream hosts with an `@` suffix:
`openrouter/deepseek-v4-flash@streamlake` or `...@streamlake,baidu`. Pinned
requests never fall back to other hosts. `bench-cache hosts <target>` lists the
host slugs that serve a target's model.
"""

from __future__ import annotations

import dataclasses
import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

# Shared across targets: deterministic-ish and cheap. We measure prompt caching,
# so output length only needs to be long enough to be a realistic history entry.
BASE_SETTINGS: ModelSettings = {"temperature": 0.0, "max_tokens": 300}


Factory = Callable[[str, list[str]], tuple[Model, ModelSettings]]
"""(model_id, pinned upstream hosts; empty means default routing) -> (model, settings)."""


@dataclass(frozen=True)
class Target:
    name: str
    description: str
    model_id: str
    factory: Factory
    only: tuple[str, ...] = ()
    """Pinned OpenRouter upstream host slugs; empty means OpenRouter's default routing."""

    def build(self) -> tuple[Model, ModelSettings]:
        return self.factory(self.model_id, list(self.only))


def _openrouter(model_id: str, only: list[str]) -> tuple[Model, ModelSettings]:
    """Model served through OpenRouter's Chat Completions endpoint.

    With default routing OpenRouter picks the upstream host and relies on sticky
    routing to keep a conversation on the same host (each host has its own cache).
    """
    model = OpenRouterModel(model_id, provider=OpenRouterProvider())
    settings: OpenRouterModelSettings = {
        **BASE_SETTINGS,
        # Reasoning traces are typically stripped from history on the next
        # turn, which changes the prefix. Keep the measured prefix exact.
        "openrouter_reasoning": {"enabled": False},
        "openrouter_usage": {"include": True},
    }
    if only:
        settings["openrouter_provider"] = {"only": only, "allow_fallbacks": False}
    return model, settings


def _openrouter_responses(model_id: str, only: list[str]) -> tuple[Model, ModelSettings]:
    """Model served through OpenRouter's OpenAI-compatible Responses endpoint (/api/v1/responses).

    Stateless: the full history is re-sent every turn (no `previous_response_id`),
    so cache hits come from upstream prefix caching, same as the chat target.
    OpenRouter-specific fields (reasoning toggle, provider routing) go in `extra_body`.
    """
    model = OpenAIResponsesModel(model_id, provider=OpenRouterProvider())
    extra_body: dict[str, object] = {"reasoning": {"enabled": False}}
    if only:
        extra_body["provider"] = {"only": only, "allow_fallbacks": False}
    settings: OpenAIResponsesModelSettings = {**BASE_SETTINGS, "extra_body": extra_body}
    return model, settings


TARGETS: dict[str, Target] = {
    t.name: t
    for t in [
        Target(
            name="openrouter/deepseek-v4-flash",
            description="DeepSeek V4 Flash via OpenRouter Chat Completions, default routing",
            model_id="deepseek/deepseek-v4-flash",
            factory=_openrouter,
        ),
        Target(
            name="openrouter-responses/deepseek-v4-flash",
            description="DeepSeek V4 Flash via OpenRouter Responses API, default routing",
            model_id="deepseek/deepseek-v4-flash",
            factory=_openrouter_responses,
        ),
    ]
}


def resolve_target(spec: str) -> Target:
    """Look up `name` or `name@host[,host...]`, returning the target with those hosts pinned."""
    base, sep, hosts = spec.partition("@")
    if base not in TARGETS:
        raise KeyError(f"unknown target {base!r}; have {sorted(TARGETS)}")
    only = tuple(h.strip() for h in hosts.split(",") if h.strip())
    if sep and not only:
        raise ValueError(f"target spec {spec!r} has '@' but no host slugs")
    return dataclasses.replace(TARGETS[base], name=spec, only=only)


def list_hosts(model_id: str) -> list[dict[str, Any]]:
    """Upstream hosts serving `model_id` on OpenRouter, with the slug to use after `@`."""
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    with urllib.request.urlopen(url, timeout=30) as resp:
        endpoints = json.load(resp)["data"]["endpoints"]
    return [
        {
            "slug": (e.get("tag") or "").split("/")[0],
            "name": e.get("provider_name"),
            "tag": e.get("tag"),
            "prompt": e["pricing"].get("prompt"),
            "cache_read": e["pricing"].get("input_cache_read"),
        }
        for e in endpoints
    ]
