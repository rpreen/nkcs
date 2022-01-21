#!/usr/bin/python3
#
# Copyright (C) 2019--2022 Richard Preen <rpreen@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""Plots experimental results."""

import os
from typing import Final, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from constants import Constants as Cons
from perf import read_data

FILE_LIST: List[str] = []  # add data file names here (without .dat)

PLOT_BESTS: Final[bool] = True  #: whether to plot the the best fitnesses
PLOT_AVERAGES: Final[bool] = False  #: whether to plot the mean fitnesses
USE_TEX: Final[bool] = False  #: whether to use texlive for plot font
CONF: Final[float] = 1  #: 1.96 = 95% confidence; 1 = standard error
ALPHA: Final[float] = 0.3  #: transparency for shading confidence bounds
MS: Final[int] = 5  #: marker size
ME: Final[int] = 2  #: mark every
LW: Final[int] = 1  #: line width
NUM_COLORS: Final[int] = 10  #: number of line colours

if USE_TEX:
    plt.rc("font", **{"family": "serif", "serif": ["Palatino"]})
    params = {"text.usetex": True, "text.latex.preamble": r"\usepackage{amstext}"}
    plt.rcParams.update(params)


def get_title() -> str:
    """Returns the title."""
    if USE_TEX:
        title = f"$N$={Cons.N} $K$={Cons.K}"
        if Cons.S > 1:
            title += f" $C$={Cons.C} $S$={Cons.S}"
    else:
        title = f"N={Cons.N} K={Cons.K}"
        if Cons.S > 1:
            title += f" C={Cons.C} S={Cons.S}"
    return title


def get_label(text: str) -> str:
    """Returns the plot label."""
    label = Cons.ACQUISITION.upper()
    if Cons.ACQUISITION != "ea":
        label += "-" + Cons.MODEL.upper()
    label += " " + text
    return label


def plot(filenames: List[str], plotname: str) -> None:
    """Plots performance from multiple sets of runs."""
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.get_cmap("tab10")
    cycler = plt.cycler(color=[cm(i / NUM_COLORS) for i in range(NUM_COLORS)])
    cycler += plt.cycler(marker=["s", "o", "^", "x", "*", "+", "X"])
    ax.set_prop_cycle(cycler)
    for data in filenames:
        evals, perf_best, perf_avg = read_data(data)
        mean_best = np.mean(perf_best, axis=0)
        mean_avg = np.mean(perf_avg, axis=0)
        if PLOT_BESTS:
            ax.plot(
                evals,
                mean_best,
                linewidth=LW,
                markersize=MS,
                markevery=ME,
                label=get_label("best"),
            )
            ax.fill_between(
                evals,
                mean_best - (CONF * stats.sem(perf_best, axis=0)),
                mean_best + (CONF * stats.sem(perf_best, axis=0)),
                alpha=ALPHA,
            )
        if PLOT_AVERAGES:
            ax.plot(
                evals,
                mean_avg,
                linewidth=LW,
                markersize=MS,
                markevery=ME,
                label=get_label("avg"),
            )
            ax.fill_between(
                evals,
                mean_avg - (CONF * stats.sem(perf_avg, axis=0)),
                mean_avg + (CONF * stats.sem(perf_avg, axis=0)),
                alpha=ALPHA,
            )
    ax.grid(linestyle="dotted", linewidth=1)
    # ax.set_ylim([3.2, 4.2])
    ax.set_xlim(xmin=0)
    ax.legend(loc="best", prop={"size": 10})
    plt.title(get_title(), fontsize=14)
    ax.set_xlabel("Evaluations", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    path: Final[str] = os.path.normpath(f"res/{plotname}.pdf")
    fig.savefig(path, bbox_inches="tight")


def stat_summary(name: str, array: np.ndarray) -> None:
    """Prints descriptive statistics summary of an array."""
    print(
        f"{name}: "
        f"MEAN={np.mean(array, axis=0)},"
        f"SD={np.std(array, axis=0)},"
        f"SE={stats.sem(array, axis=0)},"
        f"N={len(array)},"
        f"MIN={np.min(array, axis=0)},"
        f"MEDIAN={np.median(array, axis=0)}"
    )


def stat(filename1: str, filename2: str, generation: int) -> None:
    """Compares the best individuals at a specified generation."""
    _, perf_best1, _ = read_data(filename1)
    _, perf_best2, _ = read_data(filename2)
    a: np.ndarray = perf_best1[:, generation]
    b: np.ndarray = perf_best2[:, generation]
    stat_summary("A", a)
    stat_summary("B", b)
    (s, p) = stats.ranksums(a, b)
    print(f"Wilcoxon rank-sums: A vs. B: stat = {s:.5f}, p <= {p:.5f}\n")


# plots all experiments if this script is executed
if __name__ == "__main__":
    if len(FILE_LIST) > 0:
        plot(FILE_LIST, "plot")
