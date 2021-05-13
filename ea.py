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

'''Evolutionary algorithms for NKCS.'''

from copy import deepcopy
import numpy as np
from scipy.stats import norm
from constants import Constants as cons
from surrogate import Model

class Ind:
    '''A binary NKCS individual within an evolutionary algorithm.'''

    def __init__(self):
        '''Initialises a random individual.'''
        self.fitness = 0 #: fitness of the individual
        self.genome = [np.random.randint(0, 2) for i in range(cons.N)] #: genome

    def to_string(self):
        '''Returns a string representation of an individual.'''
        return str(self.genome) + ' => ' + str(self.fitness)

    def mutate(self):
        '''Mutates an individual.'''
        for i in range(cons.N):
            if np.random.uniform(low=0, high=1) < cons.P_MUT:
                if self.genome[i] == 0:
                    self.genome[i] = 1
                else:
                    self.genome[i] = 0

    def one_point_crossover(self, parent):
        '''Performs one-point crossover.'''
        if np.random.uniform(low=0, high=1) < cons.P_CROSS:
            p1 = np.random.randint(0, cons.N)
            for i in range(p1):
                self.genome[i] = parent.genome[i]

    def two_point_crossover(self, parent):
        '''Performs two-point crossover.'''
        if np.random.uniform(low=0, high=1) < cons.P_CROSS:
            p1 = np.random.randint(0, cons.N)
            p2 = np.random.randint(0, cons.N) + 1
            if p1 > p2:
                tmp = p1
                p1 = p2
                p2 = tmp
            elif p1 == p2:
                p2 += 1
            for i in range(p1, p2):
                self.genome[i] = parent.genome[i]

    def uniform_crossover(self, parent):
        '''Performs uniform crossover.'''
        if np.random.uniform(low=0, high=1) < cons.P_CROSS:
            for i in range(cons.N):
                if np.random.uniform(low=0, high=1) < 0.5:
                    self.genome[i] = parent.genome[i]

class EA:
    '''NKCS evolutionary algorithm.'''

    def __init__(self, nkcs, evals, pbest, pavg):
        '''Initialises the evolutionary algorithm.'''
        self.evals = 0 #: number of evaluations performed
        self.archive_genes = [[] for s in range(cons.S)] #: evaluated genes
        self.archive_fitness = [[] for s in range(cons.S)] #: corresponding fitnesses
        # create the initial populations
        self.pop = []
        for s in range(cons.S):
            species = [Ind() for p in range(cons.P)]
            self.pop.append(species)
        # select random initial partners
        rpartners = []
        team = []
        for s in range(cons.S):
            r = self.pop[s][np.random.randint(0, cons.P)]
            rpartners.append(r)
            team.append(r)
        # evaluate initial populations
        for s in range(cons.S):
            for p in range(cons.P):
                team[s] = self.pop[s][p]
                self.initial_eval(nkcs, team, s, evals, pbest, pavg)
            team[s] = rpartners[s]

    def print_pop(self):
        '''Prints the current population.'''
        for s in range(cons.S):
            print('Species %d' % s)
            for p in range(cons.P):
                print(self.pop[s][p].to_string())

    def print_archive(self, s):
        '''Prints the evaluated genes and fitnesses of a given species.'''
        for i in range(len(self.archive_genes[s])):
            for n in range(cons.N):
                print(str(self.archive_genes[s][i][n]), end='')
            print(',%.5f' % self.archive_fitness[s][i])

    def get_fittest(self, s):
        '''Returns the index of the best individual in a given species.'''
        best = 0
        fbest = self.pop[s][best].fitness
        for i in range(1, cons.P):
            if self.pop[s][i].fitness > fbest:
                best = i
                fbest = self.pop[s][i].fitness
        return best

    def get_best_fit(self, s):
        '''Returns the fitness of the best individual in a given species.'''
        return self.pop[s][self.get_fittest(s)].fitness

    def get_avg_fit(self, s):
        '''Returns the average fitness of a given species.'''
        avg = 0
        for i in range(cons.P):
            avg += self.pop[s][i].fitness
        avg /= cons.P
        return avg

    def initial_eval(self, nkcs, team, sp, evals, pbest, pavg):
        '''Evaluates the initial populations.'''
        # calculate team fitness
        team_fit = 0
        for s in range(cons.S):
            team_fit += nkcs.calc_fit(s, team)
        # assign fitness to the individual in this species
        team[sp].fitness = team_fit
        # archive
        self.archive_genes[sp].append(team[sp].genome) # unscaled
        self.archive_fitness[sp].append(team[sp].fitness)
        if len(self.archive_genes[sp]) > cons.MAX_ARCHIVE:
            self.archive_genes[sp].pop(0)
            self.archive_fitness[sp].pop(0)
        self.evals += 1 # total team evals performed
        if self.evals % (cons.P * cons.S) == 0:
            self.update_perf(evals, pbest, pavg)

    def eval_team(self, nkcs, team, evals, pbest, pavg):
        '''Evaluates a candidate team.'''
        # calculate team fitness
        team_fit = 0
        for s in range(cons.S):
            team_fit += nkcs.calc_fit(s, team)
        # assign fitness to each individual if best seen
        for s in range(cons.S):
            if team[s].fitness < team_fit:
                team[s].fitness = team_fit
                # archive
                self.archive_genes[s].append(team[s].genome)
                self.archive_fitness[s].append(team[s].fitness)
                if len(self.archive_genes[s]) > cons.MAX_ARCHIVE:
                    self.archive_genes[s].pop(0)
                    self.archive_fitness[s].pop(0)
        self.evals += 1 # total team evals performed
        if self.evals % (cons.P * cons.S) == 0:
            self.update_perf(evals, pbest, pavg)

    def eval(self, nkcs, sp, child, evals, pbest, pavg):
        '''Selects the best partners for a child and evaluates the team.'''
        team = []
        for s in range(cons.S):
            if s != sp:
                team.append(self.pop[s][self.get_fittest(s)])
            else:
                team.append(child)
        self.eval_team(nkcs, team, evals, pbest, pavg)

    def tournament(self, s):
        '''Returns a parent selected by tournament.'''
        best = np.random.randint(0, cons.P)
        fbest = self.pop[s][best].fitness
        for _ in range(0, cons.T_SIZE):
            competitor = np.random.randint(0, cons.P)
            fcompetitor = self.pop[s][competitor].fitness
            if fcompetitor > fbest:
                best = competitor
                fbest = fcompetitor
        return best

    def neg_tournament(self, s):
        '''Returns an individual selected by negative tournament.'''
        worst = np.random.randint(0, cons.P)
        fworst = self.pop[s][worst].fitness
        for _ in range(0, cons.T_SIZE):
            competitor = np.random.randint(0, cons.P)
            fcompetitor = self.pop[s][competitor].fitness
            if fcompetitor < fworst:
                worst = competitor
                fworst = fcompetitor
        return worst

    def create_offspring(self, p1, p2):
        '''Creates and returns a new offspring.'''
        child = deepcopy(p1)
        child.fitness = 0
        child.uniform_crossover(p2)
        child.mutate()
        return child

    def update_perf(self, evals, perf_best, perf_avg):
        '''Updates current performance tracking.'''
        best = self.get_best_fit(0)
        avg = self.get_avg_fit(0)
        for s in range(1, cons.S):
            gb = self.get_best_fit(s)
            if gb > best:
                best = gb
            avg += self.get_avg_fit(s)
        gen = int((self.evals / (cons.P * cons.S)) - 1)
        evals[gen] = self.evals
        perf_best[gen] = best
        perf_avg[gen] = avg / cons.S

    def run_cea(self, nkcs, evals, pbest, pavg):
        '''Executes a standard CEA - partnering with the best candidates.'''
        while self.evals < cons.MAX_EVALS:
            for s in range(cons.S):
                p1 = self.pop[s][self.tournament(s)]
                p2 = self.pop[s][self.tournament(s)]
                child = self.create_offspring(p1, p2)
                self.eval(nkcs, s, child, evals, pbest, pavg)
                replace = self.neg_tournament(s)
                self.pop[s][replace] = deepcopy(child)

    def run_boa(self, nkcs, evals, pbest, pavg):
        '''Executes a Bayesian optimisation assisted CEA.'''
        XI = 0.01
        while self.evals < cons.MAX_EVALS:
            for s in range(cons.S):
                model = Model()
                model.train(self.archive_genes[s], self.archive_fitness[s])
                # best of M offspring from 1 parent
                p1 = self.pop[s][self.tournament(s)]
                p2 = self.pop[s][self.tournament(s)]
                best = self.create_offspring(p1, p2)
                mu, std = model.predict(best.genome)
                # best.fitness = mu
                # best.fitness = mu + std # upper confidence bound
                mu_sample_opt = np.max(self.archive_fitness[s])
                best.fitness = 0
                if std != 0:
                    imp = mu - mu_sample_opt - XI
                    Z = imp / std
                    ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
                    best.fitness = ei
                for _ in range(1, cons.M):
                    child = self.create_offspring(p1, p2)
                    mu, std = model.predict(child.genome)
                    # child.fitness = mu
                    # child.fitness = mu + std # upper confidence bound
                    child.fitness = 0
                    if std != 0:
                        imp = mu - mu_sample_opt - XI
                        Z = imp / std
                        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
                        child.fitness = ei
                    if child.fitness > best.fitness:
                        best = child
                # evaluate best offspring
                best.fitness = 0
                self.eval(nkcs, s, best, evals, pbest, pavg)
                replace = self.neg_tournament(s)
                self.pop[s][replace] = deepcopy(best)

    def run_scea(self, nkcs, evals, pbest, pavg):
        '''Executes a surrogate-assisted CEA.'''
        model = [Model() for s in range(cons.S)]
        while self.evals < cons.MAX_EVALS:
            for s in range(cons.S):
                model[s].train(self.archive_genes[s], self.archive_fitness[s])
                # best of M offspring from 1 parent
                p1 = self.pop[s][self.tournament(s)]
                p2 = self.pop[s][self.tournament(s)]
                best = self.create_offspring(p1, p2)
                best.fitness, _ = model[s].predict(best.genome)
                for _ in range(1, cons.M):
                    child = self.create_offspring(p1, p2)
                    child.fitness, _ = model[s].predict(child.genome)
                    if child.fitness > best.fitness:
                        best = child
                # evaluate best offspring
                best.fitness = 0
                self.eval(nkcs, s, best, evals, pbest, pavg)
                replace = self.neg_tournament(s)
                self.pop[s][replace] = deepcopy(best)
