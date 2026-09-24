"""Resolve target specs into pydantic-ai models.

A target spec is `provider:model_id`, e.g. `openai:gpt-5-mini` or
`google:gemini-2.5-flash`. Providers with a factory in `FACTORIES` get
settings tuned for measuring caching. Any other provider prefix goes through
pydantic-ai's `infer_model`, so every provider pydantic-ai supports works
without code changes here.

OpenRouter specs may pin upstream hosts with an `@` suffix:
`openrouter:deepseek/deepseek-v4-flash@streamlake` or `...@streamlake,baidu`.
Pinned requests never fall back to other hosts. `bench-cache hosts <spec>`
lists the host slugs that serve a model.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

# Shared across targets: deterministic-ish and cheap. We measure prompt caching,
# so output length only needs to be long enough to be a realistic history entry.
BASE_SETTINGS: ModelSettings = {"temperature": 0.0, "max_tokens": 300}


Factory = Callable[[str, tuple[str, ...]], tuple[Model, ModelSettings]]
"""(model_id, pinned upstream hosts; empty means default routing) -> (model, settings)."""


@dataclass(frozen=True)
class Target:
    name: str
    """The spec as given, e.g. `openrouter:deepseek/deepseek-v4-flash@baidu`; labels results."""
    provider: str
    model_id: str
    hosts: tuple[str, ...] = ()
    """Pinned OpenRouter upstream host slugs; empty means OpenRouter's default routing."""

    def build(self) -> tuple[Model, ModelSettings]:
        factory = FACTORIES[self.provider].factory if self.provider in FACTORIES else _inferred(self.provider)
        return factory(self.model_id, self.hosts)


def _openrouter(model_id: str, hosts: tuple[str, ...]) -> tuple[Model, ModelSettings]:
    """Model served through OpenRouter's Chat Completions endpoint.

    With default routing OpenRouter picks the upstream host and relies on sticky
    routing to keep a conversation on the same host (each host has its own cache).
    """
    model = OpenRouterModel(model_id, provider=OpenRouterProvider())
    # mypy can't match a ModelSettings spread to the subtype's optional keys, though every key fits.
    settings: OpenRouterModelSettings = {
        **BASE_SETTINGS,  # type: ignore[typeddict-item]
        # Reasoning traces are typically stripped from history on the next
        # turn, which changes the prefix. Keep the measured prefix exact.
        "openrouter_reasoning": {"enabled": False},
        "openrouter_usage": {"include": True},
    }
    if hosts:
        settings["openrouter_provider"] = {"only": list(hosts), "allow_fallbacks": False}
    return model, settings


def _openrouter_responses(model_id: str, hosts: tuple[str, ...]) -> tuple[Model, ModelSettings]:
    """Model served through OpenRouter's OpenAI-compatible Responses endpoint (/api/v1/responses).

    Stateless: the full history is re-sent every turn (no `previous_response_id`),
    so cache hits come from upstream prefix caching, same as the chat target.
    OpenRouter-specific fields (reasoning toggle, provider routing) go in `extra_body`.
    """
    model = OpenAIResponsesModel(model_id, provider=OpenRouterProvider())
    extra_body: dict[str, object] = {"reasoning": {"enabled": False}}
    if hosts:
        extra_body["provider"] = {"only": list(hosts), "allow_fallbacks": False}
    settings: OpenAIResponsesModelSettings = {**BASE_SETTINGS, "extra_body": extra_body}  # type: ignore[typeddict-item]
    return model, settings


def _inferred(provider: str) -> Factory:
    """Any other pydantic-ai provider, with thinking off through the cross-provider setting."""

    def factory(model_id: str, hosts: tuple[str, ...]) -> tuple[Model, ModelSettings]:
        return infer_model(f"{provider}:{model_id}"), {**BASE_SETTINGS, "thinking": False}

    return factory


@dataclass(frozen=True)
class ProviderFactory:
    description: str
    factory: Factory
    pins_hosts: bool = False
    """Accepts `@host[,host...]` to pin OpenRouter upstream hosts."""


FACTORIES: dict[str, ProviderFactory] = {
    "openrouter": ProviderFactory("OpenRouter Chat Completions; OPENROUTER_API_KEY", _openrouter, pins_hosts=True),
    "openrouter-responses": ProviderFactory(
        "OpenRouter Responses API, stateless; OPENROUTER_API_KEY", _openrouter_responses, pins_hosts=True
    ),
}


def resolve_target(spec: str) -> Target:
    """Parse `provider:model_id`, plus `@host[,host...]` for providers that pin OpenRouter hosts."""
    provider, sep, model_id = spec.partition(":")
    if not sep or not provider or not model_id:
        raise ValueError(f"target {spec!r} is not provider:model, e.g. openai:gpt-5-mini")
    hosts: tuple[str, ...] = ()
    # Only split on '@' where it means hosts: other providers may use it in model ids.
    if provider in FACTORIES and FACTORIES[provider].pins_hosts:
        model_id, at, host_list = model_id.partition("@")
        hosts = tuple(h.strip() for h in host_list.split(",") if h.strip())
        if at and not hosts:
            raise ValueError(f"target {spec!r} has '@' but no host slugs")
    return Target(name=spec, provider=provider, model_id=model_id, hosts=hosts)


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
