"""A shared trunk of filler turns that splits into branches, which can split again.

`trunk = 2, branches = 4` branches at turn 2 into 4:

    A -> B -> C1
         B -> C2
         B -> C3
         B -> C4

`branches` takes one entry per level, and every branch splits again after its
`branch_turns`. `trunk = 2, branches = [3, 2]` is A -> B, B splits 3 ways
(C1..C3), and each C splits 2 ways: 1 + 1 + 3 + 6 = 11 turns. With
`branch_turns = 2` each branch runs two turns before it splits.

`branch_at` forks sets of branches from other trunk turns too, each set with
the same shape. `trunk = 5, branch_at = [3, 5], branches = 2` is A1..A5, then
two branches from A3 and two from A5.

The trunk runs first, then each set in `branch_at` order. Branches run depth
first: each runs to its leaves before its next sibling starts.

Turns at the same depth get the same filler and differ only in their tag, so
siblings share a prefix exactly up to the fork, and every turn after the first
should hit its parent's cache:

    | [session key] system | [key/t1] A | OK | [key/t2] B | OK | [key/t3] C1
    | [session key] system | [key/t1] A | OK | [key/t2] B | OK | [key/t4] C2
      \\_____________________ shared prefix: B's input _____________________/

Parameters:
    trunk          turns in the shared trunk
    branch_at      trunk turns to fork a set of branches from: an int or a list (default: the last trunk turn)
    branches       ways each split fans out: an int for one split, or a list with one entry per level
    branch_turns   turns each branch runs before it splits again or ends: an int, or one per level
    turn_tokens    approximate tokens per turn
    seed           filler word sequence
"""

from .filler import filler, ok_system
from .scenario import Expectation, Turn

system = ok_system


def _per_level(name: str, value: int | list[int], levels: int) -> list[int]:
    values = value if isinstance(value, list) else [value] * levels
    if len(values) != levels:
        raise ValueError(f"{name} has {len(values)} entries but branches has {levels} levels")
    if any(v < 1 for v in values):
        raise ValueError(f"{name} entries must be at least 1, got {values}")
    return values


def _parents(trunk: int, forks: list[int], widths: list[int], lengths: list[int]) -> list[int]:
    """The parent of each turn (0 is the bare system prompt), in depth-first order."""
    parents = list(range(trunk))

    def grow(fork: int, level: int) -> None:
        if level == len(widths):
            return
        for _ in range(widths[level]):
            prev = fork
            for _ in range(lengths[level]):
                parents.append(prev)
                prev = len(parents)
            grow(prev, level + 1)

    for fork in forks:
        grow(fork, 0)
    return parents


def turns(
    key: str,
    trunk: int = 2,
    branch_at: int | list[int] | None = None,
    branches: int | list[int] = 2,
    branch_turns: int | list[int] = 1,
    turn_tokens: int = 3000,
    seed: int = 0,
) -> list[Turn]:
    if trunk < 0:
        raise ValueError(f"trunk must be at least 0, got {trunk}")
    forks = [trunk] if branch_at is None else branch_at if isinstance(branch_at, list) else [branch_at]
    if bad := [f for f in forks if not 0 <= f <= trunk]:
        raise ValueError(f"branch_at {bad} must be trunk turns (0 to {trunk})")
    widths = _per_level("branches", branches, len(branches) if isinstance(branches, list) else 1)
    lengths = _per_level("branch_turns", branch_turns, len(widths))

    depth = {0: 0}
    result = []
    for i, parent in enumerate(_parents(trunk, forks, widths, lengths), start=1):
        depth[i] = depth[parent] + 1
        # With no trunk, later roots share only the system prompt, which is usually too short to cache.
        expect: Expectation = "hit" if parent else ("miss" if i == 1 else "any")
        result.append(Turn(filler(turn_tokens, tag=f"{key}/t{i}", seed=seed + depth[i]), expect=expect, parent=parent))
    return result
