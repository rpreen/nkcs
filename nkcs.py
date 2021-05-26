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

'''An implementation of the NKCS model for exploring aspects of coevolution.'''

import sys
import random
import numpy as np
from constants import Constants as cons

class NKCS:
    '''NKCS model.'''

    def __init__(self):
        '''Initialises a randomly generated NKCS model.'''
        self.species = [self.Species(i) for i in range(cons.S)]

    def calc_fit(self, sp, team):
        '''Returns the fitness of an individual partnered with a given team.'''
        total = 0
        N = len(team[sp].genome)
        for i in range(N):
            inputs = self.get_gene_inputs(sp, team, i)
            total += self.species[sp].gene_fit(inputs, i)
        return total / N

    def calc_team_fit(self, team):
        '''Returns the total team fitness.'''
        total = 0
        for s in range(cons.S):
            total += self.calc_fit(s, team)
        return total

    def get_gene_inputs(self, sp, team, gene_idx):
        '''Returns the inputs to a gene (including the internal state).'''
        species = self.species[sp] # species containing the gene
        inputs = [0 for i in range(species.n_gene_inputs)] # inputs to the gene
        offset = gene_idx * (species.n_gene_inputs - 1) # map offset
        map_idx = len(team[sp].genome) - cons.N # which map depends on num genes
        cnt = 0
        # internal connections
        for _ in range(cons.K):
            node = species.map[map_idx][offset + cnt]
            inputs[cnt] = team[sp].genome[node]
            cnt += 1
        # external connections
        if cons.NKCS_TOPOLOGY == 'line':
            if sp != 0:
                left = cons.S - 1 if sp - 1 < 0 else sp - 1
                map_idx = len(team[left].genome) - cons.N
                for _ in range(cons.C):
                    node = species.map[map_idx][offset + cnt]
                    inputs[cnt] = team[left].genome[node]
                    cnt += 1
            if sp != cons.S - 1:
                right = (sp + 1) % cons.S
                map_idx = len(team[right].genome) - cons.N
                for _ in range(cons.C):
                    node = species.map[map_idx][offset + cnt]
                    inputs[cnt] = team[right].genome[node]
                    cnt += 1
        elif cons.NKCS_TOPOLOGY == 'standard':
            for j in range(cons.S):
                if j != sp:
                    map_idx = len(team[j].genome) - cons.N
                    for _ in range(cons.C):
                        node = species.map[map_idx][offset + cnt]
                        inputs[cnt] = team[j].genome[node]
                        cnt += 1
        else:
            print('unsupported NKCS topology')
            sys.exit()
        # internal state
        inputs[cnt] = team[sp].genome[gene_idx]
        return inputs

    def display(self, sp):
        '''Prints a specified NKCS species.'''
        print('**********************')
        print('[%d] SPECIES:' % sp)
        print('**********************')
        self.species[sp].display()

    class Species:
        '''A species within an NKCS model.'''

        def __init__(self, sp):
            '''Initialises a species with random connectivity.'''
            X = 0 #: number of coevolving species
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
            self.n_gene_inputs = cons.K + (X * cons.C) + 1 #: n inputs to each gene
            map_length = cons.N * (self.n_gene_inputs - 1) #: connectivity length
            self.map = [np.random.randint(0, cons.N, map_length)] #: connectivity
            ftable_length = cons.N + cons.MAX_GROW
            self.ftable = [{} for i in range(ftable_length)] #: each gene's hash table
            # build species connectivity maps for varying number of genes
            for i in range(cons.MAX_GROW):
                prev_map_length = map_length
                map_length += self.n_gene_inputs - 1
                sp_map = np.resize(self.map[i], map_length)
                # add new gene's connections
                sp_map[prev_map_length :] = \
                    np.random.randint(0, cons.N + i, self.n_gene_inputs - 1)
                # wire the first connection of a random gene to the new gene
                sp_map[random.randint(0, cons.N + i)] = cons.N + i
                self.map.append(sp_map)

        def gene_fit(self, inputs, gene):
            '''Returns the fitness of an individual gene within a species.'''
            key = tuple(inputs)
            # find fitness in table
            if key in self.ftable[gene]:
                return self.ftable[gene].get(key)
            # not found, add new
            fitness = np.random.uniform(low=0, high=1)
            self.ftable[gene][key] = fitness
            return fitness

        def display(self):
            '''Prints an NKCS species.'''
            print('con: ' + str(self.map[0]))
            print('fitness table:')
            for i in range(len(self.ftable)):
                print('Gene (%d)' % i)
                print(self.ftable[i])
