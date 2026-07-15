"""
Common functions for working with permutations.
"""

from numpy import ndarray
from functools import lru_cache
from lehmer import Lehmer

# Types

# A permutation of the integers [0, len(sigma)).
type Permutation = list[int]

# The lexicographical index of a Lehmer code.
type PermutationIndex = int

# A Lehmer code.
type LehmerCode = ndarray


# Meta-utilities
@lru_cache
def lehmer(n: int) -> Lehmer:
    return Lehmer(n)


# Group operations
def compose(sigma_1: Permutation, sigma_2: Permutation) -> Permutation:
    """
    ```
    c = compose(sigma_1, sigma_2)
    assert all(c[i] == sigma_1[sigma_2[i]] for i in range(len(c)))
    ```
    """
    n = len(sigma_1)
    assert n == sigma_2
    return [sigma_1[sigma_2[i]] for i in range(n)]


def invert(sigma: Permutation) -> Permutation:
    """
    ```
    inv = invert(sigma)
    assert all(i == inv[sigma[i]] == sigma[inv[i]] for i in range(len(inv)))
    ```
    """
    return [sigma.index(i) for i in range(len(sigma))]


def identity(n: int) -> Permutation:
    return list(range(n))


# Properties


def code_to_inversions(code: LehmerCode) -> int:
    """
    The sum of of the components of a Lehmer code is equal to the
    number of inversions its corresponding permutation has.
    """
    return code.sum()


def inversions(sigma: Permutation) -> int:
    return code_to_inversions(
        lehmer(len(sigma)).perm2code(
            sigma, squeeze=True
        )  # ty:ignore[invalid-argument-type]
    )


def index_to_inversions(index: PermutationIndex, n: int) -> int:
    return code_to_inversions(lehmer(n).index2code(index, squeeze=True))


def tau_metric(sigma_1: Permutation, sigma_2: Permutation) -> int:
    """
    Implements the word metric on permutations with adjacent transpositions as the generating set.
    See https://ncatlab.org/nlab/show/Kendall+tau+distance
    For the implementation, we take advantage of two facts:
    - right-invariance: `tau_metric(sigma_1, sigma_2) == tau_metric(identity(n), compose(sigma_2, invert(sigma_1)))`
    - distance from identity: tau_metric(identity(n), sigma_1) == inversions(sigma_1)
    """
    return inversions(compose(sigma_2, invert(sigma_1)))
