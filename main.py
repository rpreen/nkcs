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

'''Main script for starting NKCS (co)evolutionary experiments.'''

import os
import sys
import warnings
import dill
from tqdm import tqdm
from constants import Constants as cons # parameters are in constants.py
from constants import get_filename

# set number of CPU threads
os.environ['OMP_NUM_THREADS'] = str(cons.NUM_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(cons.NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(cons.NUM_THREADS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cons.NUM_THREADS)
os.environ['NUMEXPR_NUM_THREADS'] = str(cons.NUM_THREADS)
warnings.filterwarnings('ignore') # surpress warnings

import numpy as np
from nkcs import NKCS
from ea import EA
from perf import save_data
from plot import plot

# results storage
N_RES = cons.F * cons.E
evals = np.zeros((N_RES, cons.G))
perf_best = np.zeros((N_RES, cons.G))
perf_avg = np.zeros((N_RES, cons.G))

r = 0 #: run counter
nkcs = [] #: NKCS landscapes
ea = [] #: EA populations

if cons.EXPERIMENT_LOAD: # reuse fitness landscapes and initial populations
    with open('experiment.pkl', 'rb') as f:
        ea = dill.load(f)
        nkcs = dill.load(f)
    if len(nkcs) != cons.F or len(ea) != N_RES:
        print('loaded experiment does not match constants')
        sys.exit()
    for _ in range(cons.F):
        for _ in range(cons.E):
            ea[r].update_perf(evals[r], perf_best[r], perf_avg[r])
            r += 1
else: # create new fitness landscapes and initial populations
    for f in range(cons.F):
        nkcs.append(NKCS())
        for _ in range(cons.E):
            ea.append(EA(nkcs[f]))
            ea[r].update_perf(evals[r], perf_best[r], perf_avg[r])
            r += 1
    if cons.EXPERIMENT_SAVE:
        with open('experiment.pkl', 'wb') as f:
            dill.dump(ea, f)

# run the experiments
r = 0
bar = tqdm(total=N_RES) #: progress bar
for f in range(cons.F): # F NKCS functions
    for e in range(cons.E): # E experiments
        if cons.ACQUISITION == 'ea':
            ea[r].run_ea(nkcs[f], evals[r], perf_best[r], perf_avg[r])
        else:
            ea[r].run_sea(nkcs[f], evals[r], perf_best[r], perf_avg[r])
        status = ('nkcs (%d) experiment (%d) complete: (%.5f, %d)' %
            (f, e, ea[r].get_best_fit(0), ea[r].get_best_length(0))))
        r += 1
        bar.set_description(status)
        bar.refresh()
        bar.update(1)
bar.close()

if cons.EXPERIMENT_SAVE:
    with open('experiment.pkl', 'a+b') as f:
        dill.dump(nkcs, f)

# write performance to a file and plot results
FILENAME = get_filename()
save_data(FILENAME, evals, perf_best, perf_avg)
if cons.PLOT:
    plot([FILENAME], FILENAME)
