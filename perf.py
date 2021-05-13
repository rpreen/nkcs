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

'''Functions for displaying and plotting NKCS EA performance.'''

import os
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy import stats
from constants import Constants as cons

def plot(filename, evals, perf_best, perf_avg):
    '''Plots EA performance'''
    plt.rc('font',**{'family':'serif','serif':['Palatino']})
    params = { 'text.usetex': True, 'text.latex.preamble': r"\usepackage{amstext}" }
    plt.rcParams.update(params)
    CONF = 1.96 # 1.96 = 95% confidence; 1 = standard error
    ALPHA = 0.3 # transparency for shading confidence bounds
    MS = 5 # marker size
    ME = 2 # mark every
    LW = 1 # line width
    NUM_COLORS = 10
    cm = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 0+1)
    ax.set_prop_cycle(color=[cm(i/NUM_COLORS) for i in range(NUM_COLORS)])
    mean_evals = np.mean(evals, axis=0)
    mean_best = np.mean(perf_best, axis=0)
    mean_avg = np.mean(perf_avg, axis=0)
    # plot bests
    ax.plot(mean_evals, mean_best, marker='o',
        linewidth=LW, markersize=MS, markevery=ME, label='Best')
    ax.fill_between(mean_evals, mean_best - (CONF * stats.sem(perf_best, axis=0)),
        mean_best + (CONF * stats.sem(perf_best, axis=0)), alpha=ALPHA)
    # plot averages
    ax.plot(mean_evals, mean_avg, marker='s',
        linewidth=LW, markersize=MS, markevery=ME, label='Avg.')
    ax.fill_between(mean_evals, mean_avg - (CONF * stats.sem(perf_avg, axis=0)),
        mean_avg + (CONF * stats.sem(perf_avg, axis=0)), alpha=ALPHA)
    ax.grid(linestyle='dotted', linewidth=1)
    #ax.set_ylim([3.2, 4.2])
    ax.set_xlim(xmin=0)
    ax.legend(loc='best', prop={'size': 10})
    TITLE = '$N$=' + str(cons.N) \
        + ' $K$=' + str(cons.K) \
        + ' $C$=' + str(cons.C) \
        + ' $S$=' + str(cons.S)
    plt.title(TITLE, fontsize=14)
    ax.set_xlabel('Evaluations', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    path = os.path.normpath('res/'+filename+'.pdf')
    fig.savefig(path, bbox_inches='tight')

def save(filename, evals, perf_best, perf_avg):
    '''Writes the results to a file.'''
    path = os.path.normpath('res/'+filename+'.dat')
    f = open(path, 'w')
    dim = cons.F * cons.E
    for g in range(cons.G):
        f.write('%d' % (evals[0][g])) # evaluations
        for r in range(dim): # best from each run
            f.write(' %f' % (perf_best[r][g]))
        for r in range(dim): # average from each run
            f.write(' %f' % (perf_avg[r][g]))
        f.write('\n')
    f.close()

def perf_display(filename, evals, perf_best, perf_avg):
    '''Writes NKCS EA performance to a file and plots the results.'''
    save(filename, evals, perf_best, perf_avg)
    if cons.PLOT:
        plot(filename, evals, perf_best, perf_avg)
