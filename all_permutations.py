from typing import MutableSequence
from lehmer import Lehmer
from math import factorial
from itertools import product
import numpy as np


def pref2perm(k: int, i: int) -> int:
    """
    Inverse of perm2pref.
    pref2perm_i(k): [n] - {i} -> [n-1]
    """
    return k if k < i else k - 1


def perm2pref(k: int, i: int) -> int:
    """
    Compose with a member of S_n to translate
    a preference list into a permutation.
    perm2pref_i(k): [n-1] -> [n] - {i}
    """
    return k if k < i else k + 1


def invert_permutation(perm: list[int]) -> list[int]:
    return [perm.index(i) for i in range(len(perm))]


def insert_at(l: list[int], i: int, n: int) -> list[int]:
    l.insert(i, n - 1)
    return l


def implies(p: bool, q: bool) -> bool:
    return not p or q


def prefers(i: int, a: int, b: int, rank: list[list[int]]) -> bool:
    """
    Does i prefer a to b?
    """
    return rank[i][a] < rank[i][b]


def test2(table: np.ndarray, perm: list[int], n: int) -> bool:
    perm_inv = invert_permutation(perm)
    rank = [
        insert_at(invert_permutation(list(row)), idx, n)
        for idx, row in enumerate(table)
    ]

    # for all i: i does not prefer perm[i] to perm_inv[i]
    # there does not exist i: i prefers perm[i] to perm_inv[i]
    for i in range(n):
        if prefers(i, perm[i], perm_inv[i], rank):
            return False

    # for all i, j: (i prefers j to perm[i]) implies (j prefers perm[j] to i)
    for i, j in product(range(n), repeat=2):
        if not implies(prefers(i, j, perm[i], rank), prefers(j, perm[j], i, rank)):
            return False

    return True


def test(table: np.ndarray, perm: list[int], n: int) -> bool:
    """
    Test that a permutation is stable according to Tan.
    """
    perm_inv = invert_permutation(perm, n)
    rank = [invert_permutation(list(row), n - 1) for row in table]

    # No one (i) prefers perm[i] to perm_inv[i]
    for i in range(n):
        # Fixed points are vacuously alright
        if i == perm[i]:
            continue
        if rank[i][pref2perm(perm[i], i)] < rank[i][pref2perm(perm_inv[i], i)]:
            return False

    # No (i) such that i prefers j to perm[i] and yet j doesn't prefer perm[j] to i
    for i, j in product(range(n), repeat=2):
        # i ranks itself last
        if i == j:
            continue
        elif (
            (i == perm[i])
            or (rank[i][pref2perm(j, i)] < rank[i][pref2perm(perm[i], i)])
        ) and (
            j != perm[j] and rank[j][pref2perm(perm[j], j)] >= rank[j][pref2perm(i, j)]
        ):
            return False

    return True


type Result = tuple[tuple[int, ...], int, bool]


def main(n: int) -> list[Result]:
    lc = Lehmer(n)
    lc_pred = Lehmer(n - 1)

    results: list[tuple[tuple[int, ...], int, bool]] = []

    for preference_indices in product(range(factorial(n - 1)), repeat=n):
        table = lc_pred.decode(np.array(preference_indices))
        for perm_idx in range(factorial(n)):
            perm: list[int] = list(lc.decode(perm_idx, squeeze=True))
            result = test2(table, perm, n)
            # print(result)
            results.append((preference_indices, perm_idx, result))

    return results


if __name__ == "__main__":
    def test_all():
        """
        Test correctness.
        """
        from fractions import Fraction

        results = main(4)

        assert len(results) == 31_104

        def prob(pi: int) -> Fraction:
            sub_results = [r for r in results if r[1] == pi]
            return Fraction(sum(int(r[2]) for r in sub_results), len(sub_results))

        print(
            prob(7),
            prob(0),
            prob(9),
            prob(3),
        )

        # [2^2]
        assert prob(7) == Fraction(233, 648)
        # [1^4]
        assert prob(0) == 0
        # [2^1 1^2]
        assert prob(1) == 0
        # TODO: Both asserts fail: probability zero
        # [4^1]
        assert prob(9) == Fraction(25, 1296)
        # [1^1, 1^3]
        assert prob(3) == Fraction(1, 216)

    def display():
        import networkx as nx
        from networkx.algorithms import bipartite
        import matplotlib.pyplot as plt

        n = 4
        results = main(n)
        B = nx.Graph()
        B.add_nodes_from(product(range(factorial(n - 1)), repeat=n), bipartite=0)
        B.add_nodes_from(range(factorial(n)), bipartite=1)
        B.add_edges_from([r[:2] for r in results if r[2]])
        nx.draw(
            B,
            pos=nx.bipartite_layout(
                B, nodes=product(range(factorial(n - 1)), repeat=n)
            ),
        )
        plt.show()

    test_all()
