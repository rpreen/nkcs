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

'''Parameters for NKCS coevolutionary experimentation.'''

class Constants:
    '''Global constants.'''
    F = 10 #: number of different NKCS experiments to run
    E = 10 #: number of experiments per NKCS to run
    G = 20 #: number of generations to run genetic algorithm
    N = 20 #: number of genes (nodes/characteristics)
    K = 2 #: number of connections to other (internal) genes
    C = 2 #: number of external genes affecting each gene
    S = 2 #: number of species
    P = 20 #: number of individuals in each species population
    T_SIZE = 3 #: size of selection and replacement tournament
    P_MUT = 1 / N #: per gene probability of mutation
    P_CROSS = 0.8 #: probability of performing crossover
    M = 100 #: number of children per parent to test via model
    MAX_ARCHIVE = 5000 #: max evaluated individuals for model training
    H = 20 #: number of hidden neurons for MLP
    N_MODELS = 1 #: number of surrogate models (for averaging, and std dev)
    PLOT = True #: plot graph
    ALGORITHM = 'boa' #: algorithm = {'ea', 'boa', 'sea'}
    MAX_EVALS = P * S * G #: number of evaluations per experiment
    NKCS_TOPOLOGY = 'line' #: topology = {'line', 'full'}
    NUM_THREADS = 8 #: number of CPU threads for model building
