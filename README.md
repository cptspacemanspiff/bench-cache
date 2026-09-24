# bench-cache

Benchmarks how well LLM providers reuse cached prompts across multi-turn conversations. Built on pydantic-ai.

```sh
uv run bench-cache list
uv run bench-cache run linear -t openrouter-responses:deepseek/deepseek-v4-flash -p n_turns=10 -p turn_tokens=3000
uv run bench-cache run linear -t openai:gpt-5-mini -t google:gemini-2.5-flash
uv run bench-cache run branch -t openai:gpt-5-mini -p trunk=2 -p branches=3      # A -> B, then B -> C1, C2, C3
uv run bench-cache run branch -t openai:gpt-5-mini -p trunk=2 -p branches=3,2    # ... and each C splits in 2 again
uv run bench-cache hosts openrouter:deepseek/deepseek-v4-flash      # upstream host slugs + cache pricing
uv run bench-cache run linear -t openrouter:deepseek/deepseek-v4-flash@streamlake -t openrouter:deepseek/deepseek-v4-flash@baidu
uv run bench-cache run suites/basic.toml -T targets/default.toml -n 5   # a suite of cases against a list of targets
```

A target is `provider:model_id`. `openrouter` (Chat Completions) and `openrouter-responses` (Responses API) have their own settings in `targets.py`. Any other prefix goes to pydantic-ai's `infer_model`, so every provider pydantic-ai supports works, with that provider's usual API key variable (e.g. `OPENAI_API_KEY`). Its SDK must be installed, and only the `openai` extra is by default. Providers that cache only at explicit breakpoints (Anthropic, Bedrock) have no factory yet, so they will show no cache reads.

Without a suffix, OpenRouter picks the upstream host. Each host has its own cache, so an unpinned result depends partly on routing. Add `@host` (or `@host1,host2`) to an OpenRouter target to pin it to those host slugs, with no fallback to other hosts.

Each run prints a per-turn table and writes three files to `./results` (change with `--out-dir`), all named after the run:

- `<run>.jsonl`: every turn of every conversation
- `<run>_cached.png`: cached tokens per turn, one panel per target. Each conversation is one line, and in each turn's column the runs sit side by side (run 1 leftmost), so every run's hit or miss stays visible. Each run's first turn is marked hollow if it read nothing from cache. A red diamond means it read cache that could only have come from another run. The dashed line is the most each turn could have reused (its parent turn's input). A branching scenario is drawn as a tree: branches at the same depth share a column, each in its own colour, fanning out from the turn they branched from.
- `<run>_hits.png`: the same hits as a grid, one row per target and one box per turn. Each box fills from the bottom by the share of conversations whose cache grew on that turn, with the count below it (e.g. `1/2`) and the target's total on the right. In a branching scenario the boxes are laid out as a tree: each branch gets its own row under its trunk, with a connector down from the turn it branched from.

## Scenarios

Both built-in scenarios send filler text under a system prompt that tells the model to reply only "OK", so the history is almost entirely the filler you asked for. They differ only in the shape of the conversation.

**`linear`**: one conversation, each turn continuing the last. `n_turns` turns of `turn_tokens` each, or give `turn_tokens` a list to size each turn separately.

**`branch`**: a shared trunk that splits into branches, which can split again.

| params | conversation |
|---|---|
| `trunk = 2, branches = 4` | A -> B, then B -> C1, C2, C3, C4 (branch at turn 2, width 4) |
| `trunk = 2, branches = 2, branch_turns = 3` | A -> B, then two branches of 3 turns each |
| `trunk = 2, branches = [3, 2]` | A -> B, B splits 3 ways, and each of those splits 2 ways (11 turns) |
| `trunk = 2, branches = [3, 2], branch_turns = [2, 1]` | as above, but the first-level branches run 2 turns before splitting |
| `trunk = 5, branch_at = [3, 5], branches = 2` | A1..A5, then two branches from A3 and two from A5 |

`branch_at` lists the trunk turns to fork a set of branches from (default: the last trunk turn); every set has the same shape, and the sets run in the order listed, after the trunk. `branches` has one entry per level of splitting. `branch_turns` is how many turns each branch runs before it splits again or ends: one number for every level, or a list with one per level. Branches run depth first, so each reaches its leaves before its next sibling starts. Turns at the same depth get the same filler and differ only in a tag at the start, so siblings share a prefix exactly up to the fork, and every turn after the first should hit its parent's cache.

## Suites

A suite file fixes *what* runs: a list of named cases, each a scenario with its parameters. *Where* it runs and *how many times* are chosen per run, so the same suite can be rerun later, against other targets, and at a quick `-n 1` or a thorough `-n 20`.

```toml
# suites/basic.toml
description = "Linear multi-turn and branching conversations"

[[case]]
name = "linear"
scenario = "linear"
params = { n_turns = 10, turn_tokens = 3000 }

[[case]]
name = "branch-t2-w4"          # branch at turn 2 into 4
scenario = "branch"
params = { trunk = 2, branches = 4 }

[[case]]
name = "nested"                # branch at turn 2 into 3, then each of those into 2
scenario = "branch"
params = { trunk = 2, branches = [3, 2] }

[[case]]
name = "width"
scenario = "branch"
params = { trunk = 2 }
sweep = { branches = [2, 4, 8] }   # one case per value: width.branches=2, ...
```

- `name` defaults to the scenario name. Names must be unique in the suite, since each case's results file is named after it.
- `params` override the scenario's defaults. Lists are TOML lists (`turn_tokens = [2048, 8000]`). `-p` doesn't apply to a suite, so everything that shapes the run stays in the file.
- `sweep` expands one `[[case]]` into the cartesian product of its lists, with each combination appended to the name.
- The whole suite is loaded and checked (scenarios exist, parameters are known, turn structure is valid) before any request is sent.

Targets come from `-t` and from any number of `-T` targets files, which hold only a list:

```toml
# targets/default.toml
targets = [
    "openrouter:deepseek/deepseek-v4-flash",
    "openrouter:deepseek/deepseek-v4-flash@streamlake",
]
```

Cases run in order, and each runs on every target `-n` times. A suite run writes to `results/<stamp>_<suite>/`:

- `<case>.jsonl`, `<case>_cached.png` and `<case>_hits.png` for each case, the same files as a single-scenario run. They are written as each case finishes, so an interrupted run keeps the finished cases. Each JSONL row also records its `case`.
- `suite.toml`: a copy of the suite file as it was run.
- `run.json`: the targets, repeats, turn delay, bench-cache version, and every case's full parameters with defaults filled in, so a later change to a scenario's defaults can't silently change what an old run meant.

## Layout

- Scenarios: built-in ones are modules in the package, registered in `BUILTIN_SCENARIOS` in `scenario.py`, e.g. `src/bench_cache/linear.py`. Hand-written prompt sequences go in `./prompts` (change with `--prompts-dir`), outside the code, one Python file per scenario (create the folder when you add the first one). If both have a scenario with the same name, the `prompts/` file wins. Either kind defines:
  - `system(key, ...) -> str`
  - `turns(key, ...) -> list[Turn]`, where `Turn(prompt, expect="miss" | "hit" | "any", parent=None)` comes from `bench_cache.scenario`

  A turn continues from the previous turn's history by default. Set `parent` to an earlier turn's number to branch the conversation there instead (`0` starts again from just the system prompt).

  `key` is a fresh UUID for every conversation. Put it into f-strings to choose which prefixes are new. For example, placing it at the very start of `system()` guarantees that turn 1 can't hit the cache.

  Keyword arguments with defaults on these functions become scenario parameters. Override them with `-p name=value`, e.g. `-p n_turns=10 -p turn_tokens=3000`. A comma-separated value becomes a list, e.g. `-p turn_tokens=2048,8000`. `bench_cache.filler.filler(n_tokens, tag=..., seed=...)` returns deterministic padding of roughly `n_tokens` tokens. It starts with `[tag]`, so a key-derived tag makes each block's start a deterministic point to break the prefix. `filler.ok_system(key)` is the "always reply OK" system prompt the built-in scenarios use, so the model's replies add almost nothing to the history.
- `src/bench_cache/suite.py`: loads suite files (expanding sweeps and validating every case) and targets files.
- `src/bench_cache/targets.py`: parses target specs and builds a pydantic-ai model and its settings. `FACTORIES` holds the providers with their own settings. `openrouter:` targets call OpenRouter's Chat Completions endpoint (`OpenRouterModel`). `openrouter-responses:` targets call its Responses endpoint (`OpenAIResponsesModel`) statelessly, re-sending the full history each turn. All other providers get the shared settings with `thinking` off.
- `src/bench_cache/runner.py`: runs the turns in order, each continuing its parent turn's message history, and records each response's `RequestUsage`.
- `src/bench_cache/cli.py`: the CLI, table output and JSONL writer.
- `src/bench_cache/plot.py`: renders a results JSONL file as both PNGs. It works out hits from the cached-token counts, so older result files chart correctly too.

## Columns

| column  | meaning                                                                   |
|---------|---------------------------------------------------------------------------|
| input   | prompt tokens for this request                                            |
| cached  | prompt tokens served from cache (`cache_read_tokens`)                     |
| write   | tokens written to the cache (only for providers that report it)           |
| hit%    | cached / input                                                            |
| cost $  | billed cost if the provider reports it (OpenRouter Chat Completions), otherwise estimated from [genai-prices](https://github.com/pydantic/genai-prices), `-` if the model is unknown |
| upstream | the host that served the request, where the router reports it (OpenRouter Chat Completions only) |
| from    | the turn this one continued from; shown only when the conversation branches |
| reuse%  | cached / the previous (parent) turn's input, i.e. how much of the reusable prefix was reused |

Each conversation ends with a summary: the effective hit rate (all cached / all input) and the share of reusable prefix tokens that were actually reused. When there is more than one run, the same summary is also printed per target across all repeats.

A `hit` turn passes only if it read more from cache than the previous turn (its parent, when the conversation branches). The history only grows, so a working cache reads more every turn. Reading the same or less, e.g. 500 → 1000 → 500, means part of the cached prefix was lost, so that turn (turn 3 here) is a miss.

## Development

```sh
uv sync
uv run pre-commit install          # run the checks on every commit
uv run pre-commit run --all-files  # or run them by hand
```

The hooks check that `uv.lock` matches `pyproject.toml`, then run ruff (lint and format), mypy (strict) and deptry (declared vs. imported dependencies). Every tool comes from the dev group, so versions are pinned by `uv.lock`. CI (`.github/workflows/ci.yml`) runs the same hooks on pushes to `main` and on pull requests. Tool settings are in `pyproject.toml`.
