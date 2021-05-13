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
import warnings
from constants import Constants as cons # parameters are in constants.py

# set number of CPU threads
os.environ['OMP_NUM_THREADS'] = str(cons.NUM_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(cons.NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(cons.NUM_THREADS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cons.NUM_THREADS)
os.environ['NUMEXPR_NUM_THREADS'] = str(cons.NUM_THREADS)
warnings.filterwarnings('ignore') # surpress warnings

from tqdm import tqdm
import numpy as np
from nkcs import NKCS
from ea import EA
from perf import get_filename, save_data
from plot import plot

# results storage
N_RES = cons.F * cons.E
evals = np.zeros((N_RES, cons.G))
perf_best = np.zeros((N_RES, cons.G))
perf_avg = np.zeros((N_RES, cons.G))

bar = tqdm(total=N_RES) #: progress bar
r = 0 #: run counter
for f in range(cons.F): # F NKCS functions
    nkcs = NKCS()
    for e in range(cons.E): # E experiments
        ea = EA(nkcs, evals[r], perf_best[r], perf_avg[r])
        if cons.ALGORITHM == 'ea':
            ea.run_cea(nkcs, evals[r], perf_best[r], perf_avg[r])
        elif cons.ALGORITHM == 'boa':
            ea.run_boa(nkcs, evals[r], perf_best[r], perf_avg[r])
        else:
            ea.run_scea(nkcs, evals[r], perf_best[r], perf_avg[r])
        status = ('nkcs (%d) experiment (%d) complete: (%.5f)' %
            (f, e, ea.get_best_fit(0)))
        r += 1
        bar.set_description(status)
        bar.refresh()
        bar.update(1)
bar.close()

# write performance to a file and plot results
filename = get_filename()
save_data(filename, evals, perf_best, perf_avg)
if cons.PLOT:
    plot([filename], filename)
