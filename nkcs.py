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

'''An implementation of the NKCS model for exploring aspects of coevolution.'''

from __future__ import annotations
import sys
from typing import List, Final, Dict
import numpy as np
from constants import Constants as cons

class NKCS:
    '''NKCS model.'''

    def __init__(self) -> None:
        '''Initialises a randomly generated NKCS model.'''
        self.species: List[NKCS.Species] = [self.Species(i) for i in range(cons.S)]

    def __calc_fit(self, sp: int, team: List[np.ndarray]) -> float:
        '''Returns the fitness of an individual partnered with a given team.'''
        total: float = 0
        for i in range(cons.N):
            inputs = self.__get_gene_inputs(sp, team, i)
            total += self.species[sp].gene_fit(inputs, i)
        return total / cons.N

    def calc_team_fit(self, team: List[np.ndarray]) -> float:
        '''Returns the total team fitness.'''
        total: float = 0
        for s in range(cons.S):
            total += self.__calc_fit(s, team)
        return total

    def __get_gene_inputs(self, sp: int, team: List[np.ndarray], gene_idx: int) -> np.ndarray:
        '''Returns the inputs to a gene (including the internal state).'''
        species: NKCS.Species = self.species[sp] # species containing the gene
        inputs: np.ndarry = np.zeros(species.n_gene_inputs) # inputs to the gene
        offset: int = gene_idx * (species.n_gene_inputs - 1) # map offset
        cnt: int = 0
        # internal connections
        for _ in range(cons.K):
            node = species.map[offset + cnt]
            inputs[cnt] = team[sp][node]
            cnt += 1
        # external connections
        if cons.NKCS_TOPOLOGY == 'line':
            if sp != 0:
                left = cons.S - 1 if sp - 1 < 0 else sp - 1
                for _ in range(cons.C):
                    node = species.map[offset + cnt]
                    inputs[cnt] = team[left][node]
                    cnt += 1
            if sp != cons.S - 1:
                right = (sp + 1) % cons.S
                for _ in range(cons.C):
                    node = species.map[offset + cnt]
                    inputs[cnt] = team[right][node]
                    cnt += 1
        elif cons.NKCS_TOPOLOGY == 'standard':
            for j in range(cons.S):
                if j != sp:
                    for _ in range(cons.C):
                        node = species.map[offset + cnt]
                        inputs[cnt] = team[j][node]
                        cnt += 1
        else:
            print('unsupported NKCS topology')
            sys.exit()
        # internal state
        inputs[cnt] = team[sp][gene_idx]
        return inputs

    def display(self, sp: int) -> None:
        '''Prints a specified NKCS species.'''
        print('**********************')
        print('[%d] SPECIES:' % sp)
        print('**********************')
        self.species[sp].display()

    class Species:
        '''A species within an NKCS model.'''

        def __init__(self, sp: int) -> None:
            '''Initialises a species with random connectivity.'''
            X: int = 0 #: number of coevolving species
            if cons.S > 1:
                if cons.NKCS_TOPOLOGY == 'line':
                    if sp in (0, cons.S - 1):
                        X = 1
                    else:
                        X = 2
                elif cons.NKCS_TOPOLOGY == 'standard':
                    X = cons.S - 1
                else:
                    print('unsupported NKCS topology')
                    sys.exit()
            self.n_gene_inputs: Final[int] = cons.K + (X * cons.C) + 1 #: n inputs to each gene
            map_length: Final[int] = cons.N * (self.n_gene_inputs - 1) #: connectivity length
            self.map: np.ndarray = np.random.randint(0, cons.N, map_length) #: connectivity
            self.ftable: List[Dict[tuple, float]] = \
                [{} for i in range(cons.N)] #: each gene's hash table

        def gene_fit(self, inputs: np.ndarray, gene: int) -> float:
            '''Returns the fitness of an individual gene within a species.'''
            # find fitness in table
            key = tuple(inputs)
            fit = self.ftable[gene].get(key)
            if fit is None: # not found, add new
                fitness = np.random.uniform(low=0, high=1)
                self.ftable[gene][key] = fitness
            else:
                fitness = fit
            return fitness

        def display(self) -> None:
            '''Prints an NKCS species.'''
            print('con: ' + str(self.map))
            print('fitness table:')
            for i in range(len(self.ftable)):
                print('Gene (%d)' % i)
                print(self.ftable[i])
