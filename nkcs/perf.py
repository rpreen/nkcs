#!/usr/bin/python3
#
# Copyright (C) 2019--2024 Richard Preen <rpreen@gmail.com>
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

"""Functions for reading and writing results to a file."""

import csv
import os
from typing import Final, Tuple

import numpy as np
from constants import Constants as Cons
from constants import read_constants, save_constants


def save_data(
    filename: str,
    evals: np.ndarray,
    perf_best: np.ndarray,
    perf_avg: np.ndarray,
) -> None:
    """Write the results to a data file."""
    path: Final[str] = os.path.normpath(f"res/{filename}.dat")
    with open(path, "w", encoding="utf-8") as fp:
        save_constants(fp)
        dim: Final[int] = Cons.F * Cons.E
        for g in range(Cons.G):
            fp.write(f"{evals[0][g]}")  # evaluations
            for r in range(dim):  # best from each run
                fp.write(f",{perf_best[r][g]}")
            for r in range(dim):  # average from each run
                fp.write(f",{perf_avg[r][g]}")
            fp.write("\n")


def read_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the results from a data file."""
    path: Final[str] = os.path.normpath(f"res/{filename}.dat")
    with open(path, "r", encoding="utf-8") as csvfile:
        archive = csv.reader(csvfile, delimiter=",")
        # read constants from the header row
        row = next(archive)
        read_constants(row)
        # data rows
        n_res = Cons.F * Cons.E
        evals = np.zeros(Cons.G)
        perf_best = np.zeros((n_res, Cons.G))
        perf_avg = np.zeros((n_res, Cons.G))
        for g, row in enumerate(archive):
            evals[g] = int(float(row[0]))
            for col in range(n_res):
                perf_best[col][g] = float(row[col + 1])
            for col in range(n_res):
                perf_avg[col][g] = float(row[n_res + col + 1])
    return evals, perf_best, perf_avg
