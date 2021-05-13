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
    G = 10 #: number of generations to run genetic algorithm
    N = 20 #: number of genes (nodes/characteristics)
    K = 2 #: number of connections to other (internal) genes
    C = 2 #: number of external genes affecting each gene
    S = 2 #: number of species
    P = 20 #: number of individuals in each species population
    T_SIZE = 3 #: size of selection and replacement tournament
    P_MUT = 1 / N #: per gene probability of mutation
    P_CROSS = 0.8 #: probability of performing crossover
    M = 1000 #: number of children per parent to test via model
    MAX_ARCHIVE = 5000 #: max evaluated individuals for model training
    H = 20 #: number of hidden neurons for MLP
    N_MODELS = 1 #: number of surrogate models (for averaging, and std dev)
    PLOT = True #: plot graph
    ALGORITHM = 'ea' #: algorithm = {'ea', 'boa', 'sea'}
    MAX_EVALS = P * S * G #: number of evaluations per experiment
    NKCS_TOPOLOGY = 'line' #: topology = {'line', 'full'}
    NUM_THREADS = 8 #: number of CPU threads for model building

def get_filename():
    '''Returns a file name based on the parameters.'''
    filename = Constants.ALGORITHM \
        + 'f' + str(Constants.F) \
        + 'e' + str(Constants.E) \
        + 'g' + str(Constants.G) \
        + 'p' + str(Constants.P) \
        + 's' + str(Constants.S) \
        + 'n' + str(Constants.N) \
        + 'k' + str(Constants.K) \
        + 'c' + str(Constants.C) \
        + 'tsize' + str(Constants.T_SIZE) \
        + 'pmut' + str(Constants.P_MUT) \
        + 'pcross' + str(Constants.P_CROSS)
    if Constants.ALGORITHM != 'ea':
        filename += 'm' + str(Constants.M) \
        + 'h' + str(Constants.H) \
        + 'maxarchive' + str(Constants.MAX_ARCHIVE) \
        + 'nmodels' + str(Constants.N_MODELS)
    return filename

def save_constants(f):
    '''Writes constants to a data file.'''
    f.write('%s,' % Constants.ALGORITHM)
    f.write('%d,' % Constants.F)
    f.write('%d,' % Constants.E)
    f.write('%d,' % Constants.G)
    f.write('%d,' % Constants.P)
    f.write('%d,' % Constants.S)
    f.write('%d,' % Constants.N)
    f.write('%d,' % Constants.K)
    f.write('%d,' % Constants.C)
    f.write('%d,' % Constants.T_SIZE)
    f.write('%f,' % Constants.P_MUT)
    f.write('%f,' % Constants.P_CROSS)
    f.write('%d,' % Constants.M)
    f.write('%d,' % Constants.H)
    f.write('%d,' % Constants.MAX_ARCHIVE)
    f.write('%d,' % Constants.N_MODELS)
    f.write('%s' % Constants.NKCS_TOPOLOGY)
    f.write('\n')

def read_constants(row):
    '''Reads constants from a row.'''
    Constants.ALGORITHM = str(row[0])
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
    Constants.M = int(row[12])
    Constants.H = int(row[13])
    Constants.MAX_ARCHIVE = int(row[14])
    Constants.N_MODELS = int(row[15])
    Constants.NKCS_TOPOLOGY = str(row[16])
