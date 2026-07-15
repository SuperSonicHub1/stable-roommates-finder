import matching
import json
from tqdm import tqdm
from matching import Player
from dataclasses import dataclass
from all_permutations import test, table_to_rank

import random
from math import factorial, perm

from lehmer import Lehmer
from matching.games import StableRoommates
import numpy as np


def random_prefs(n: int) -> tuple[int, ...]:
    n_pred_fac = factorial(n - 1)
    pref_indices = tuple(random.randrange(n_pred_fac) for _ in range(n))
    return pref_indices


def prefs_to_table(prefs: tuple[int, ...], lc_pred: Lehmer) -> np.ndarray:
    return lc_pred.decode(np.array(prefs))


def prefs_to_game(prefs: tuple[int, ...], lc_pred: Lehmer) -> StableRoommates:
    player_prefs = {
        i: [j if j < i else j + 1 for j in lc_pred.decode(index, squeeze=True).tolist()]
        for i, index in enumerate(prefs)
    }
    return StableRoommates.create_from_dictionary(player_prefs)


@dataclass
class SteadyTableResult:
    prefs: tuple[int, ...]
    matching: int


def find_steady_table(n: int) -> SteadyTableResult:
    """
    Find a steady table on `n` participants:
    a preference table that has a stable matching.
    """

    lc_pred = Lehmer(n - 1)
    lc = Lehmer(n)

    while True:
        prefs = random_prefs(n)
        game = prefs_to_game(prefs, lc_pred)
        result: dict[Player, Player] = game.solve()
        if game.check_stability():
            processed: dict[int, int] = {k.name: v.name for k, v in result.items()}
            return SteadyTableResult(
                prefs,
                lc.encode(np.array([processed[i] for i in range(n)]), squeeze=True),
            )


def test_all_perms(prefs: tuple[int, ...]) -> dict[int, bool]:
    n = len(prefs)
    lc = Lehmer(n)
    lc_pred = Lehmer(n - 1)
    rank = table_to_rank(prefs_to_table(prefs, lc_pred), n)
    return {
        perm_idx: test(rank, lc.decode(perm_idx, squeeze=True).tolist(), n)
        for perm_idx in tqdm(range(factorial(n)))
    }


if __name__ == "__main__":
    result = find_steady_table(10)
    perms2stable = test_all_perms(result.prefs)
    assert perms2stable[result.matching]

    filename = f"stable-roommates-finder-data/search_steady_tables/stable_permutations__{"_".join(str(pref) for pref in result.prefs)}.json"
    with open(filename, "w") as f:
        json.dump(
            dict(
                prefs=result.prefs,
                matching=result.matching,
                stability=[k for k, v in perms2stable.items() if v],
            ),
            f,
        )
