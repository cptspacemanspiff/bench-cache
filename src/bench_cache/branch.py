"""A shared trunk of filler turns, then branches that each continue from the trunk's last turn.

With the defaults the conversation is A -> B -> C, then B -> D:

    | [session key] system | [key/t1] A | OK | [key/t2] B | OK | [key/b1/t3] C
    | [session key] system | [key/t1] A | OK | [key/t2] B | OK | [key/b2/t3] D
      \\_____________________ shared prefix: B's input _____________________/

Branches at the same depth get the same filler and differ only in their tag, so
their prefixes split exactly where the branch starts. Every branch's first turn
should hit the trunk's cache, including after earlier branches have run. The
branches run one after another, each to completion.

Parameters:
    n_trunk        shared turns before the branch point
    n_branches     branches continuing from the trunk's last turn
    branch_turns   turns in each branch
    turn_tokens    approximate tokens per turn
    seed           filler word sequence
"""

from .filler import filler
from .ok_filler import system
from .scenario import Turn

__all__ = ["system", "turns"]


def turns(
    key: str, n_trunk: int = 2, n_branches: int = 2, branch_turns: int = 1, turn_tokens: int = 3000, seed: int = 0
) -> list[Turn]:
    trunk = [
        Turn(filler(turn_tokens, tag=f"{key}/t{i}", seed=seed + i), expect="miss" if i == 1 else "hit")
        for i in range(1, n_trunk + 1)
    ]
    branches = [
        Turn(
            filler(turn_tokens, tag=f"{key}/b{b}/t{i}", seed=seed + i),
            expect="hit",
            parent=n_trunk if i == n_trunk + 1 else None,
        )
        for b in range(1, n_branches + 1)
        for i in range(n_trunk + 1, n_trunk + branch_turns + 1)
    ]
    return trunk + branches
