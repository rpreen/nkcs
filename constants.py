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

'''Parameters for NKCS coevolutionary experimentation.'''

from typing import List, TextIO

class Constants:
    '''Global constants.'''
    F: int = 10 #: number of different NKCS experiments to run
    E: int = 10 #: number of experiments per NKCS to run
    G: int = 10 #: number of generations to run genetic algorithm
    N: int = 20 #: number of genes (nodes/characteristics)
    K: int = 2 #: number of connections to other (internal) genes
    C: int = 2 #: number of external genes affecting each gene
    S: int = 2 #: number of species
    P: int = 20 #: number of individuals in each species population
    T_SIZE: int = 3 #: size of selection and replacement tournament
    P_MUT: float = 1 / N #: per gene probability of mutation
    P_CROSS: float = 0.8 #: probability of performing crossover
    REPLACE: str = 'worst' #: replace = {'worst', 'tournament'}
    M: int = 1000 #: number of children per parent to test via model
    MAX_ARCHIVE: int = 5000 #: max evaluated individuals for model training
    H: int = 20 #: number of hidden neurons for MLP
    N_MODELS: int = 10 #: number of surrogate models (for averaging, and std dev)
    PLOT: bool = True #: plot graph
    ACQUISITION: str = 'ea' #: acquisition = {'ea', 'ei', 'uc', 'pi', 'mean'}
    MODEL: str = 'gp' #: surrogate model = {'gp', 'mlp', 'svr', 'linear', 'tree', 'gradient'}
    MAX_EVALS: int = P * S * G #: number of evaluations per experiment
    NKCS_TOPOLOGY: str = 'standard' #: topology = {'line', 'standard'}
    NUM_THREADS: int = 8 #: number of CPU threads for model building
    EXPERIMENT_SAVE: bool = False #: whether to save landscapes and initial populations
    EXPERIMENT_LOAD: bool = False #: whether to load landscapes and initial populations

def cons_to_string() -> str:
    '''Returns a string representation of the constants.'''
    string: str = '%s' % (Constants.NKCS_TOPOLOGY) \
        + '_f%d' % (Constants.F) \
        + '_e%d' % (Constants.E) \
        + '_g%d' % (Constants.G) \
        + '_p%d' % (Constants.P) \
        + '_s%d' % (Constants.S) \
        + '_n%d' % (Constants.N) \
        + '_k%d' % (Constants.K) \
        + '_c%d' % (Constants.C) \
        + '_tsize%d' % (Constants.T_SIZE) \
        + '_pmut%f' % (Constants.P_MUT) \
        + '_pcross%f' % (Constants.P_CROSS) \
        + '_replace%s' % (Constants.REPLACE) \
        + '_evals%d' % (Constants.MAX_EVALS)
    if Constants.ACQUISITION != 'ea':
        string += '_m%d' % (Constants.M) \
        + '_h%d' % (Constants.H) \
        + '_maxa%d' % (Constants.MAX_ARCHIVE) \
        + '_nmodels%d' % (Constants.N_MODELS) \
        + '_model%s' % (Constants.MODEL)
    return string

def save_constants(fp: TextIO) -> None:
    '''Writes constants to a data file.'''
    fp.write('%s,' % Constants.ACQUISITION)
    fp.write('%d,' % Constants.F)
    fp.write('%d,' % Constants.E)
    fp.write('%d,' % Constants.G)
    fp.write('%d,' % Constants.P)
    fp.write('%d,' % Constants.S)
    fp.write('%d,' % Constants.N)
    fp.write('%d,' % Constants.K)
    fp.write('%d,' % Constants.C)
    fp.write('%d,' % Constants.T_SIZE)
    fp.write('%f,' % Constants.P_MUT)
    fp.write('%f,' % Constants.P_CROSS)
    fp.write('%s,' % Constants.REPLACE)
    fp.write('%d,' % Constants.M)
    fp.write('%d,' % Constants.H)
    fp.write('%d,' % Constants.MAX_ARCHIVE)
    fp.write('%d,' % Constants.N_MODELS)
    fp.write('%s,' % Constants.NKCS_TOPOLOGY)
    fp.write('%s' % Constants.MODEL)
    fp.write('\n')

def read_constants(row: List[str]) -> None:
    '''Reads constants from a row.'''
    Constants.ACQUISITION = str(row[0])
    Constants.F = int(row[1])
    Constants.E = int(row[2])
    Constants.G = int(row[3])
    Constants.P = int(row[4])
    Constants.S = int(row[5])
    Constants.N = int(row[6])
    Constants.K = int(row[7])
    Constants.C = int(row[8])
    Constants.T_SIZE = int(row[9])
    Constants.P_MUT = float(row[10])
    Constants.P_CROSS = float(row[11])
    Constants.REPLACE = str(row[12])
    Constants.M = int(row[13])
    Constants.H = int(row[14])
    Constants.MAX_ARCHIVE = int(row[15])
    Constants.N_MODELS = int(row[16])
    Constants.NKCS_TOPOLOGY = str(row[17])
    Constants.MODEL = str(row[18])
