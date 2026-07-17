from matplotlib.figure import Figure
from itertools import chain
from tqdm import tqdm
from collections import defaultdict
from perturb_stable_tables import PerturbResults
import json
from pathlib import Path
from typing import Sequence, Optional
from search_steady_tables import prefs_to_table
import numpy as np
import matplotlib.pyplot as plt
from permutations import tau_metric, PermutationIndex, lehmer, Preferences, identity
from scipy.optimize import curve_fit


def prefs_metric(
    pref_indices_1: Preferences,
    pref_indices_2: Preferences,
    n_pred: int,
) -> int:
    """
    The word metric of a product group is the product 1-metric of
    the word metric of the product group's components.
    """
    lc_pred = lehmer(n_pred)
    return sum(
        tau_metric(sigma_1.tolist(), sigma_2.tolist())
        for sigma_1, sigma_2 in zip(
            prefs_to_table(pref_indices_1, lc_pred),
            prefs_to_table(pref_indices_2, lc_pred),
        )
    )


def analyze_json(path: Path):
    with path.open() as f:
        data: PerturbResults = json.load(f)
    if "n" not in data:
        data["n"] = len(data["prefs"])
    if "N" not in data:
        data["N"] = len(data["tests"])

    n = data["n"]
    n_pred = n - 1

    prefs = data["prefs"]
    distances = [
        prefs_metric(prefs, perturbation, n_pred)
        for perturbation in tqdm(data["perturbations"])
    ]
    max_dist = max(distances)

    distance_count: defaultdict[int, int] = defaultdict(int)
    distance_stable_count: defaultdict[int, int] = defaultdict(int)
    distance_count[0] = 1
    distance_stable_count[0] = 1

    for distance, stable in zip(distances, data["tests"]):
        distance_count[distance] += 1
        distance_stable_count[distance] += stable

    return data, max_dist, distance_stable_count, distance_count


def plot_analysis(
    data: PerturbResults,
    max_dist: int,
    distance_stable_count: dict[int, int],
    distance_count: dict[int, int],
    popt_exp: list[float],
) -> Figure:
    fig = plt.figure()
    x = list(range(max_dist + 1))
    y_actual = [
        (
            distance_stable_count[dist] / distance_count[dist]
            if distance_count[dist]
            else 0
        )
        for dist in x
    ]
    y_expected = [exponential(x, *popt_exp) for x in x]
    plt.plot(x, y_actual, label="Actual")
    plt.plot(x, y_expected, label=f"Expected (exp([a]={popt_exp}))")
    plt.xlabel("Distance from original steady table")
    plt.ylabel("Pr[matching stable | distance]")
    plt.title(f"Stability likelihood for {data['prefs']}")
    plt.suptitle(f"n={data["n"]}; N={data["N"]}")
    plt.legend()
    return fig


def exponential(x, a):
    return np.exp(-a * x)


def fit_likelihood_exponential(
    data: PerturbResults,
    max_dist: int,
    distance_stable_count: dict[int, int],
    distance_count: dict[int, int],
):
    x = list(range(max_dist + 1))
    y_actual = [
        (
            distance_stable_count[dist] / distance_count[dist]
            if distance_count[dist]
            else 0
        )
        for dist in x
    ]
    return curve_fit(exponential, x, y_actual, p0=(1,))


if __name__ == "__main__":

    def test_metric():
        from search_steady_tables import random_prefs
        from perturb_stable_tables import perturb_table
        from tqdm import tqdm

        n = 10
        n_pred = n - 1
        lc_pred = lehmer(n - 1)
        # lc = lehmer(n)
        for _ in range(100):
            # 0 on equality
            prefs = random_prefs(n)
            assert prefs_metric(prefs, prefs, n_pred) == 0

            perturbation = perturb_table(prefs, lc_pred)
            # Positivity
            ltr = prefs_metric(prefs, perturbation, n_pred)
            assert ltr > 0
            # Symmetric
            assert ltr == prefs_metric(perturbation, prefs, n_pred)
            # Triangle inequality
            perturbation_2 = perturb_table(prefs, lc_pred)
            assert ltr <= prefs_metric(prefs, perturbation_2, n_pred) + prefs_metric(
                perturbation_2, perturbation, n_pred
            )

    def test_perturbation():
        from perturb_stable_tables import apply_transposition

        n = 10
        lc_pred = lehmer(n - 1)
        for i in range(n - 1):
            # The adjacent transpositions are indeed the generators of this word metric
            assert (
                tau_metric(
                    identity(9),
                    lc_pred.decode(
                        apply_transposition(0, i, lc_pred), squeeze=True
                    ).tolist(),
                )
                == 1
            )

    base = Path("stable-roommates-finder-data")
    read_base = base / "perturb_stable_tables"
    write_base = base / "perturb_analysis"

    for path in tqdm(read_base.glob("*.json")):
        name = path.stem
        save_path = write_base / f"{name}.png"
        if save_path.exists():
            continue

        data, max_dist, distance_stable_count, distance_count = analyze_json(path)
        popt_exp, _ = fit_likelihood_exponential(
            data, max_dist, distance_stable_count, distance_count
        )
        fig = plot_analysis(
            data, max_dist, distance_stable_count, distance_count, popt_exp
        )
        fig.set_figwidth(11)
        fig.set_figheight(8.5)
        fig.savefig(write_base / f"{name}.png", dpi=300)
        plt.close(fig)
