#!/usr/bin/python3.8
#
# Copyright (C) 2019--2021 Richard Preen <rpreen@gmail.com>
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

'''Functions for reading and writing results to a file.'''

import os
import csv
from typing import Tuple, Final
import numpy as np
from constants import Constants as cons
from constants import save_constants, read_constants

def save_data(filename: str, evals: np.ndarray, perf_best: np.ndarray,
        perf_avg: np.ndarray) -> None:
    '''Writes the results to a data file.'''
    path: Final[str] = os.path.normpath('res/'+filename+'.dat')
    fp = open(path, 'w')
    save_constants(fp)
    dim: Final[int] = cons.F * cons.E
    for g in range(cons.G):
        fp.write('%d' % (evals[0][g])) # evaluations
        for r in range(dim): # best from each run
            fp.write(',%f' % (perf_best[r][g]))
        for r in range(dim): # average from each run
            fp.write(',%f' % (perf_avg[r][g]))
        fp.write('\n')
    fp.close()

def read_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Reads the results from a data file.'''
    path: Final[str] = os.path.normpath('res/'+filename+'.dat')
    with open(path, 'r') as csvfile:
        archive = csv.reader(csvfile, delimiter=',')
        # read constants from the header row
        row = next(archive)
        read_constants(row)
        # data rows
        n_res = cons.F * cons.E
        evals = np.zeros(cons.G)
        perf_best = np.zeros((n_res, cons.G))
        perf_avg = np.zeros((n_res, cons.G))
        g = 0
        for row in archive:
            evals[g] = int(row[0])
            for col in range(n_res):
                perf_best[col][g] = float(row[col + 1])
            for col in range(n_res):
                perf_avg[col][g] = float(row[n_res + col + 1])
            g += 1
    return evals, perf_best, perf_avg
