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

"""Main script for starting NKCS (co)evolutionary experiments."""

from __future__ import annotations

from typing import Final

import dill
import numpy as np
from tqdm import tqdm

from constants import Constants as Cons  # parameters are in constants.py
from constants import cons_to_string
from ea import EA
from nkcs import NKCS
from perf import save_data
from plot import plot

if __name__ == "__main__":
    """Run experiment.."""

    # results storage
    n_res: Final[int] = Cons.F * Cons.E
    evals: np.ndarray = np.zeros((n_res, Cons.G))
    perf_best: np.ndarray = np.zeros((n_res, Cons.G))
    perf_avg: np.ndarray = np.zeros((n_res, Cons.G))

    r: int = 0  # run counter
    nkcs: list[NKCS] = []  # NKCS landscapes
    ea: list[EA] = []  # EA populations

    if Cons.EXPERIMENT_LOAD:  # reuse fitness landscapes and initial populations
        with open("experiment.pkl", "rb") as fp:
            ea = dill.load(fp)
            nkcs = dill.load(fp)
        if len(nkcs) != Cons.F or len(ea) != n_res:
            raise Exception("loaded experiment does not match constants")
        for _ in range(Cons.F):
            for _ in range(Cons.E):
                ea[r].update_perf(evals[r], perf_best[r], perf_avg[r])
                r += 1
    else:  # create new fitness landscapes and initial populations
        for f in range(Cons.F):
            nkcs.append(NKCS())
            for _ in range(Cons.E):
                ea.append(EA())
                ea[r].run_initial(nkcs[f])
                ea[r].update_perf(evals[r], perf_best[r], perf_avg[r])
                r += 1

    if Cons.EXPERIMENT_SAVE:  # save initial populations
        with open("experiment.pkl", "wb") as fp:
            dill.dump(ea, fp)

    # run the experiments
    r = 0
    bar = tqdm(total=n_res)  # progress bar
    for f in range(Cons.F):  # F NKCS functions
        for e in range(Cons.E):  # E experiments
            if Cons.ACQUISITION == "ea":
                ea[r].run_ea(nkcs[f], evals[r], perf_best[r], perf_avg[r])
            else:
                ea[r].run_sea(nkcs[f], evals[r], perf_best[r], perf_avg[r])
            best_fit: Final[float] = ea[r].get_best_fit(0)
            status = f"nkcs {f} experiment {e} complete: {best_fit:.5f}"
            r += 1
            bar.set_description(status)
            bar.refresh()
            bar.update(1)
    bar.close()

    if Cons.EXPERIMENT_SAVE:  # save fitness landscapes
        with open("experiment.pkl", "a+b") as fp:
            dill.dump(nkcs, fp)

    # write performance to a file and plot results
    filename: Final[str] = cons_to_string()
    save_data(filename, evals, perf_best, perf_avg)
    if Cons.PLOT:
        plot([filename], filename)
