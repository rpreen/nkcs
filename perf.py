#!/usr/bin/python3
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
import numpy as np
from constants import Constants as cons
from constants import save_constants, read_constants

def save_data(filename, evals, perf_best, perf_avg, len_best):
    '''Writes the results to a data file.'''
    path = os.path.normpath('res/'+filename+'.dat')
    f = open(path, 'w')
    save_constants(f)
    dim = cons.F * cons.E
    for g in range(cons.G):
        f.write('%d' % (evals[0][g])) # evaluations
        for r in range(dim): # best from each run
            f.write(',%f' % (perf_best[r][g]))
        for r in range(dim): # average from each run
            f.write(',%f' % (perf_avg[r][g]))
        for r in range(dim): # length of the best from each run
            f.write(',%d' % (len_best[r][g]))
        f.write('\n')
    f.close()

def read_data(filename):
    '''Reads the results from a data file.'''
    path = os.path.normpath('res/'+filename+'.dat')
    with open(path, 'r') as csvfile:
        archive = csv.reader(csvfile, delimiter=',')
        # read constants from the header row
        row = next(archive)
        read_constants(row)
        # data rows
        N_RES = cons.F * cons.E
        evals = np.zeros(cons.G)
        perf_best = np.zeros((N_RES, cons.G))
        perf_avg = np.zeros((N_RES, cons.G))
        len_best = np.zeros((N_RES, cons.G))
        g = 0
        for row in archive:
            evals[g] = int(row[0])
            for col in range(N_RES):
                perf_best[col][g] = float(row[col + 1])
            for col in range(N_RES):
                perf_avg[col][g] = float(row[N_RES + col + 1])
            for col in range(N_RES):
                len_best[col][g] = int(row[(2 * N_RES) + col + 1])
            g += 1
    return evals, perf_best, perf_avg, len_best
