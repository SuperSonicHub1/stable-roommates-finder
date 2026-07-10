from tqdm import tqdm
import random
from lehmer import Lehmer
from search_steady_tables import find_steady_table, prefs_to_table
from all_permutations import test, table_to_rank


def apply_transposition(index: int, i: int, lc_pred: Lehmer) -> int:
    """
    Multiply the permutation at index with the transposition (i, i + 1).
    Follows https://www.mathe2.uni-bayreuth.de/frib/KERBER/h00/node30.html
    """
    n_prev = lc_pred.n
    code = lc_pred.index2code(index, squeeze=True)
    i_succ = i % n_prev
    code_i, code_i_succ = code[i], code[i_succ]
    if code_i > code_i_succ:
        code[i_succ] = (code_i - 1) % n_prev
        code[i] = code_i_succ
    else:
        code[i_succ] = code_i
        code[i] = (code_i_succ + 1) % n_prev
    return lc_pred.code2index(code, squeeze=True)


def perturb_table(prefs: tuple[int, ...], lc_pred: Lehmer):
    n_prev = lc_pred.n
    n = n_prev + 1
    perm_to_modify = random.randrange(n)
    modified = apply_transposition(
        prefs[perm_to_modify], random.randrange(n_prev), lc_pred
    )
    return prefs[:perm_to_modify] + (modified,) + prefs[perm_to_modify + 1 :]

if __name__ == "__main__":
    n = 10
    N = 1_000_000

    lc_pred = Lehmer(n - 1)
    lc = Lehmer(n)
    result = find_steady_table(n)
    perm = lc.decode(result.matching, squeeze=True).tolist()
    perturbations = [perturb_table(result.prefs, lc_pred) for _ in range(N)]
    tests = [
        test(table_to_rank(prefs_to_table(prefs, lc_pred), n), perm, n)
        for prefs in tqdm(perturbations)
    ]
    
