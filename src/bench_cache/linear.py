"""A single conversation of N filler turns, each continuing the last.

Layout of the request on turn n (| marks where a key starts a new, uncached prefix):

    | [session key] system | [key/t1] filler | OK | [key/t2] filler | OK | ... | [key/tn] filler
      \\______________ turn n-1 input, should be cached on turn n ______________/

The session key at the start of the system prompt makes turn 1 cold. Each
filler block starts with its own key derived from the session key, so a
scenario can break the prefix at an exact block boundary by changing only
that block's tag.

Parameters:
    n_turns       number of turns
    turn_tokens   approximate tokens per turn: an int for every turn, or a list
                  with one size per turn, which overrides n_turns
                  (`-p turn_tokens=2048,8000` is two turns)
    seed          filler word sequence
"""

from .filler import filler, ok_system
from .scenario import Turn

system = ok_system


def turns(key: str, n_turns: int = 10, turn_tokens: int | list[int] = 3000, seed: int = 0) -> list[Turn]:
    sizes = turn_tokens if isinstance(turn_tokens, list) else [turn_tokens] * n_turns
    return [
        Turn(filler(size, tag=f"{key}/t{i}", seed=seed + i), expect="miss" if i == 1 else "hit")
        for i, size in enumerate(sizes, start=1)
    ]
