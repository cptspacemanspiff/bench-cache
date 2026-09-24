# bench-cache

Benchmarks how well LLM providers reuse cached prompts across multi-turn conversations. Built on pydantic-ai.

```sh
uv run bench-cache list
uv run bench-cache run ok_filler -t openrouter-responses/deepseek-v4-flash -p n_turns=10 -p turn_tokens=3000
uv run bench-cache hosts openrouter/deepseek-v4-flash      # upstream host slugs + cache pricing
uv run bench-cache run ok_filler -t openrouter/deepseek-v4-flash@streamlake -t openrouter/deepseek-v4-flash@baidu
```

Without a suffix, OpenRouter picks the upstream host. Each host has its own cache, so an unpinned result depends partly on routing. Add `@host` (or `@host1,host2`) to a target name to pin it to those OpenRouter host slugs, with no fallback to other hosts.

Needs `OPENROUTER_API_KEY`. Each run prints a per-turn table and writes three files to `results/`, all named after the run:

- `<run>.jsonl`: every turn of every conversation
- `<run>_cached.png`: cached tokens per turn, one panel per target and one line per conversation, with each turn marked hit or miss. The dashed line is the most each turn could have reused (the previous turn's input).
- `<run>_hits.png`: the same hits as a grid, one row per conversation and one column per turn. A filled cell means the cache grew. Totals are shown per conversation (right) and per turn across conversations (below).

## Layout

- Scenarios: built-in ones are modules in the package, registered in `BUILTIN_SCENARIOS` in `scenario.py`, e.g. `src/bench_cache/ok_filler.py`. Hand-written prompt sequences go in `prompts/`, outside the code, one Python file per scenario (create the folder when you add the first one). If both have a scenario with the same name, the `prompts/` file wins. Either kind defines:
  - `system(key, ...) -> str`
  - `turns(key, ...) -> list[Turn]`, where `Turn(prompt, expect="miss" | "hit" | "any")` comes from `bench_cache.scenario`

  `key` is a fresh UUID for every conversation. Put it into f-strings to choose which prefixes are new. For example, placing it at the very start of `system()` guarantees that turn 1 can't hit the cache.

  Keyword arguments with defaults on these functions become scenario parameters. Override them with `-p name=value`, e.g. `-p n_turns=10 -p turn_tokens=3000`. A comma-separated value becomes a list, e.g. `-p turn_tokens=2048,8000`. `bench_cache.filler.filler(n_tokens, tag=..., seed=...)` returns deterministic padding of roughly `n_tokens` tokens. It starts with `[tag]`, so a key-derived tag makes each block's start a deterministic point to break the prefix. `ok_filler` uses it with an "always reply OK" system prompt, so the model's replies add almost nothing to the history.
- `src/bench_cache/targets.py`: the provider/model registry. Each target builds a pydantic-ai model and its settings. `openrouter/...` targets call OpenRouter's Chat Completions endpoint (`OpenRouterModel`). `openrouter-responses/...` targets call its Responses endpoint (`OpenAIResponsesModel`) statelessly, re-sending the full history each turn. The Responses endpoint doesn't name the upstream host, and pydantic-ai drops its `usage.cost` field, so both columns show `-`.
- `src/bench_cache/runner.py`: runs the turns in order, carrying the message history forward, and records each response's `RequestUsage`.
- `src/bench_cache/cli.py`: the CLI, table output and JSONL writer.
- `src/bench_cache/plot.py`: renders a results JSONL file as both PNGs. It works out hits from the cached-token counts, so older result files chart correctly too.

## Columns

| column  | meaning                                                                   |
|---------|---------------------------------------------------------------------------|
| input   | prompt tokens for this request                                            |
| cached  | prompt tokens served from cache (`cache_read_tokens`)                     |
| write   | tokens written to the cache (only for providers that report it)           |
| hit%    | cached / input                                                            |
| reuse%  | cached / the previous turn's input, i.e. how much of the reusable prefix was reused |

Each conversation ends with a summary: the effective hit rate (all cached / all input) and the share of reusable prefix tokens that were actually reused. When there is more than one run, the same summary is also printed per target across all repeats.

A `hit` turn passes only if it read more from cache than the previous turn. The history only grows, so a working cache reads more every turn. Reading the same or less, e.g. 500 → 1000 → 500, means part of the cached prefix was lost, so that turn (turn 3 here) is a miss.
