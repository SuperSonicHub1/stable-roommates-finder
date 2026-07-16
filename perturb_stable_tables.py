from typing import TypedDict
from permutations import Preferences, PermutationIndex
import json
from tqdm import tqdm
import random
from lehmer import Lehmer
from search_steady_tables import find_steady_table, prefs_to_table
from all_permutations import test, table_to_rank


def apply_transposition(index: PermutationIndex, i: int, lc_pred: Lehmer) -> int:
    """
    Multiply the permutation at index with the transposition (i, i + 1).
    Follows https://www.mathe2.uni-bayreuth.de/frib/KERBER/h00/node30.html
    """
    n_pred = lc_pred.n
    code = lc_pred.index2code(index, squeeze=True)
    i_succ = i % n_pred
    code_i, code_i_succ = code[i], code[i_succ]
    if code_i > code_i_succ:
        code[i_succ] = (code_i - 1) % n_pred
        code[i] = code_i_succ
    else:
        code[i_succ] = code_i
        code[i] = (code_i_succ + 1) % n_pred
    return lc_pred.code2index(code, squeeze=True)


def perturb_table(prefs: Preferences, lc_pred: Lehmer) -> Preferences:
    prefs = tuple(prefs)
    n_prev = lc_pred.n
    n = n_prev + 1
    perm_to_modify = random.randrange(n)
    modified = apply_transposition(
        prefs[perm_to_modify], random.randrange(n_prev), lc_pred
    )
    return prefs[:perm_to_modify] + (modified,) + prefs[perm_to_modify + 1 :]


def generate_perturbations(
    prefs: Preferences, N: int, lc_pred: Lehmer
) -> list[Preferences]:
    choices = [prefs]
    for _ in tqdm(range(N)):
        choices.append(perturb_table(random.choice(choices), lc_pred))

    deduped = set(choices)
    deduped.remove(prefs)
    return list(deduped)


class PerturbResults(TypedDict):
    n: int
    N: int
    prefs: Preferences
    matching: PermutationIndex
    perturbations: list[Preferences]
    tests: list[bool]


if __name__ == "__main__":
    n = 10
    N = 1_000_000

    lc_pred = Lehmer(n - 1)
    lc = Lehmer(n)
    result = find_steady_table(n)
    matching_perm = lc.decode(result.matching, squeeze=True).tolist()
    perturbations = generate_perturbations(result.prefs, N, lc_pred)
    tests = [
        test(table_to_rank(prefs_to_table(prefs, lc_pred), n), matching_perm, n)
        for prefs in tqdm(perturbations)
    ]
    print(f"{sum(tests)}/{len(tests)}")
    with open(
        f"stable-roommates-finder-data/perturb_stable_tables/stable_tables__{"_".join(map(str, result.prefs))}.json",
        "w",
    ) as f:
        json.dump(
            dict(
                n=n,
                N=len(tests),
                prefs=result.prefs,
                matching=result.matching,
                perturbations=perturbations,
                tests=tests,
            ),
            f,
        )
