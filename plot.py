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

'''Plots experimental results.'''

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from constants import Constants as cons
from perf import read_data

FILE_LIST = [ ] # add data file names here (without .dat)

PLOT_BESTS = True #: whether to plot the the best fitnesses
PLOT_AVERAGES = False #: whether to plot the mean fitnesses
USE_TEX = False #: whether to use texlive for plot font
CONF = 1 #: 1.96 = 95% confidence; 1 = standard error
ALPHA = 0.3 #: transparency for shading confidence bounds
MS = 5 #: marker size
ME = 2 #: mark every
LW = 1 #: line width
NUM_COLORS = 10 #: number of line colours

if USE_TEX:
    plt.rc('font',**{'family':'serif','serif':['Palatino']})
    params = { 'text.usetex': True, 'text.latex.preamble': r"\usepackage{amstext}" }
    plt.rcParams.update(params)

def get_title():
    '''Returns the title.'''
    if USE_TEX:
        return '$N$=' + str(cons.N) \
            + ' $K$=' + str(cons.K) \
            + ' $C$=' + str(cons.C) \
            + ' $S$=' + str(cons.S)
    return 'N=' + str(cons.N) \
        + ' K=' + str(cons.K) \
        + ' C=' + str(cons.C) \
        + ' S=' + str(cons.S)

def get_label(l):
    '''Returns the plot label.'''
    L = cons.ACQUISITION.upper()
    if cons.ACQUISITION != 'ea':
        L += '-' + cons.MODEL.upper()
    L += ' ' + l
    return L

def plot(filenames, plotname):
    '''Plots performance from multiple sets of runs.'''
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.get_cmap('tab10')
    cycler = plt.cycler(color=[cm(i/NUM_COLORS) for i in range(NUM_COLORS)])
    cycler += plt.cycler(marker=['s', 'o', '^', 'x', '*', '+', 'X'])
    ax.set_prop_cycle(cycler)
    for data in filenames:
        evals, perf_best, perf_avg = read_data(data)
        mean_best = np.mean(perf_best, axis=0)
        mean_avg = np.mean(perf_avg, axis=0)
        if PLOT_BESTS:
            ax.plot(evals, mean_best,
                linewidth=LW, markersize=MS, markevery=ME, label=get_label('best'))
            ax.fill_between(evals, mean_best - (CONF * stats.sem(perf_best, axis=0)),
                mean_best + (CONF * stats.sem(perf_best, axis=0)), alpha=ALPHA)
        if PLOT_AVERAGES:
            ax.plot(evals, mean_avg,
                linewidth=LW, markersize=MS, markevery=ME, label=get_label('avg'))
            ax.fill_between(evals, mean_avg - (CONF * stats.sem(perf_avg, axis=0)),
                mean_avg + (CONF * stats.sem(perf_avg, axis=0)), alpha=ALPHA)
    ax.grid(linestyle='dotted', linewidth=1)
    #ax.set_ylim([3.2, 4.2])
    ax.set_xlim(xmin=0)
    ax.legend(loc='best', prop={'size': 10})
    plt.title(get_title(), fontsize=14)
    ax.set_xlabel('Evaluations', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    path = os.path.normpath('res/'+str(plotname)+'.pdf')
    fig.savefig(path, bbox_inches='tight')

# plots all experiments if this script is executed
if __name__ == '__main__':
    if len(FILE_LIST) > 0:
        plot(FILE_LIST, 'plot')
