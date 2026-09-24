"""Deterministic filler text of an approximate token size, for padding prompts.

Words are drawn from a fixed list of short, common English words, which most
BPE tokenizers encode as one token each (with the leading space). This makes
`n_tokens` a close estimate across providers. Check each turn's measured
`input` column for the exact count.
"""

from __future__ import annotations

import random

WORDS_PER_LINE = 16

# Short common words; with a leading space most tokenizers encode each as one token.
_VOCAB = """
time year people way day man thing woman life child world school state family
student group country problem hand part place case week company system program
question work government number night point home water room mother area money
story fact month lot right study book eye job word business issue side kind head
house service friend father power hour game line end member law car city name
president team minute idea kid body information back parent face others level
office door health person art war history party result change morning reason
research girl guy moment air teacher force education foot boy age policy process
music market sense nation plan college interest death experience effect use class
control care field development role effort rate heart drug show leader light voice
wife police mind price report decision son view relationship town road arm
difference value building action model season society tax director position player
record paper space ground form event official matter center couple site project
activity star table need court oil situation cost industry figure street image
phone data picture practice piece land product doctor wall patient worker news
test movie north love support technology step baby computer type attention film
tree source organization hair window evidence population green blue red
""".split()


def filler(n_tokens: int, *, tag: str = "", seed: int = 0) -> str:
    """Return about `n_tokens` tokens of deterministic filler.

    `tag` is written first, so the block starts with it. Pass a run key or a key
    derived from it to make the block unique per run and deterministically break
    the prefix at that point. `seed` picks the word sequence, and the same
    (n_tokens, seed) always produces the same words.
    """
    rng = random.Random(seed)
    words = [rng.choice(_VOCAB) for _ in range(max(n_tokens, 0))]
    lines = [" ".join(words[i : i + WORDS_PER_LINE]) for i in range(0, len(words), WORDS_PER_LINE)]
    head = f"[{tag}]\n" if tag else ""
    return head + "\n".join(lines)
