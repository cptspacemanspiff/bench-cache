"""Deterministic filler text of an approximate token size, and the system prompt
that goes with it. Every built-in scenario uses both; they differ only in the
shape of the conversation.

The filler is lorem ipsum: sentences of words drawn from the classic passage,
in a seeded order. It reads as placeholder text to every model, where random
English words can look like an obfuscated message: Anthropic's safety
classifier stopped Claude Sonnet 5.5 responses to the earlier word-salad
filler (`stop_reason: refusal`), and Claude Code retried each one, doubling
those turns' input. The Latin-like words take more than one token each, about
1.4 with OpenAI's o200k tokenizer, so `n_tokens` is converted to a word count
with `TOKENS_PER_WORD`. Other tokenizers differ (cl100k: about 1.55); check each
turn's measured `input` column for the exact count.
"""

from __future__ import annotations

import random

SENTENCES_PER_LINE = 4
# Measured with tiktoken's o200k_base on this generator's output (1.38-1.40).
TOKENS_PER_WORD = 1.4

# The words of the standard lorem ipsum passage, each once.
_VOCAB = """
lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud
exercitation ullamco laboris nisi aliquip ex ea commodo consequat duis aute irure
in reprehenderit voluptate velit esse cillum fugiat nulla pariatur excepteur sint
occaecat cupidatat non proident sunt culpa qui officia deserunt mollit anim id est
laborum
""".split()  # noqa: SIM905 -- a word block reads better than a 60-item list literal


def _sentences(rng: random.Random, n_words: int) -> list[str]:
    """Sentences of 6-14 words, capitalised and full-stopped, half with a comma."""
    out = []
    left = n_words
    while left > 0:
        k = min(left, rng.randint(6, 14))
        left -= k
        words = [rng.choice(_VOCAB) for _ in range(k)]
        if k > 7 and rng.random() < 0.5:
            words[rng.randint(2, k - 3)] += ","
        out.append(" ".join(words).capitalize() + ".")
    return out


def filler(n_tokens: int, *, tag: str = "", seed: int = 0) -> str:
    """Return about `n_tokens` tokens of deterministic lorem ipsum.

    `tag` is written first, so the block starts with it. Pass a run key or a key
    derived from it to make the block unique per run and deterministically break
    the prefix at that point. `seed` picks the word sequence, and the same
    (n_tokens, seed) always produces the same text.
    """
    rng = random.Random(seed)
    sentences = _sentences(rng, round(max(n_tokens, 0) / TOKENS_PER_WORD))
    lines = [" ".join(sentences[i : i + SENTENCES_PER_LINE]) for i in range(0, len(sentences), SENTENCES_PER_LINE)]
    head = f"[{tag}]\n" if tag else ""
    return head + "\n".join(lines)


def ok_system(key: str) -> str:
    """A system prompt telling the model to reply only "OK", so its replies add almost
    nothing to the history. The session key comes first, so turn 1 can't hit the cache."""
    return f"""[session {key}]

This conversation is an automated test of prompt caching. User messages contain
meaningless filler text. Do not read, analyze, summarize, or comment on it.
Reply to every message with exactly: OK"""
