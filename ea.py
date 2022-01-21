#!/usr/bin/python3
#
# Copyright (C) 2019--2022 Richard Preen <rpreen@gmail.com>
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

"""Evolutionary algorithms for NKCS."""

from __future__ import annotations

import random
from copy import deepcopy
from operator import attrgetter
from typing import Final

import numpy as np

import surrogate
from constants import Constants as Cons
from nkcs import NKCS


class Ind:
    """A binary NKCS individual within an evolutionary algorithm."""

    def __init__(self) -> None:
        """Initialises a random individual."""
        self.fitness: float = 0  #: fitness
        self.genome: np.ndarray = np.random.randint(2, size=Cons.N)  #: genome

    def to_string(self) -> str:
        """Returns a string representation of an individual."""
        return ",".join([str(gene) for gene in self.genome]) + f",{self.fitness}"

    def mutate(self) -> None:
        """Mutates an individual."""
        for i in range(len(self.genome)):
            if np.random.uniform(low=0, high=1) < Cons.P_MUT:
                if self.genome[i] == 0:
                    self.genome[i] = 1
                else:
                    self.genome[i] = 0

    def uniform_crossover(self, parent: Ind) -> None:
        """Performs uniform crossover."""
        for i in range(len(self.genome)):
            if np.random.uniform(low=0, high=1) < 0.5:
                self.genome[i] = parent.genome[i]


class EA:
    """NKCS evolutionary algorithm."""

    def __init__(self) -> None:
        """Initialises the evolutionary algorithm."""
        self.evals: int = 0  #: number of evaluations performed
        self.archive: list[list[Ind]] = [[] for _ in range(Cons.S)]  #: archive
        self.pop: list[list[Ind]] = []  #: current species populations

    def run_initial(self, nkcs: NKCS) -> None:
        """Generates new random initial populations."""
        # create the initial populations
        for _ in range(Cons.S):
            species = [Ind() for _ in range(Cons.P)]
            self.pop.append(species)
        # select random initial partners
        rpartners: list[Ind] = []
        team: list[Ind] = []
        for s in range(Cons.S):
            r = self.pop[s][np.random.randint(0, Cons.P)]
            rpartners.append(r)
            team.append(r)
        # evaluate initial populations
        for s in range(Cons.S):
            for p in range(Cons.P):
                team[s] = self.pop[s][p]
                genomes = np.asarray([ind.genome for ind in team])
                team[s].fitness = nkcs.calc_team_fit(genomes)
                self.__update_archive(s, team[s])
                self.evals += 1
            team[s] = rpartners[s]

    def __eval_team(self, nkcs: NKCS, team: list[Ind]) -> None:
        """Evaluates a candidate team.
        Assigns fitness to each individual if it's the best seen."""
        genomes = np.asarray([ind.genome for ind in team])
        team_fit: Final[float] = nkcs.calc_team_fit(genomes)
        for s in range(Cons.S):
            team[s].fitness = max(team[s].fitness, team_fit)
        self.evals += 1  # total team evals performed

    def __get_team_best(self, s: int, child: Ind) -> list[Ind]:
        """Returns a team assembled with the best partners for a child."""
        team: list[Ind] = []
        for sp in range(Cons.S):
            if sp != s:
                team.append(self.get_best(sp))
            else:
                team.append(child)
        return team

    def __update_archive(self, s: int, p: Ind) -> None:
        """Adds an evaluated individual to the species archive."""
        self.archive[s].append(p)
        if len(self.archive[s]) > Cons.MAX_ARCHIVE:
            self.archive[s].pop(0)

    def update_perf(
        self, evals: list[int], perf_best: np.ndarray, perf_avg: np.ndarray
    ):
        """Updates current performance tracking."""
        if self.evals % (Cons.P * Cons.S) == 0:
            g: Final[int] = int((self.evals / (Cons.P * Cons.S)) - 1)
            evals[g] = self.evals
            perf_best[g] = np.max([self.get_best_fit(s) for s in range(Cons.S)])
            perf_avg[g] = np.mean([self.get_avg_fit(s) for s in range(Cons.S)])

    def __create_offspring(self, p1: Ind, p2: Ind) -> Ind:
        """Creates and returns a new offspring."""
        child = deepcopy(p1)
        child.fitness = 0
        if np.random.uniform(low=0, high=1) < Cons.P_CROSS:
            child.uniform_crossover(p2)
        child.mutate()
        return child

    def __add_offspring(self, s: int, child: Ind) -> None:
        """Adds an offspring to the population and archive."""
        if Cons.REPLACE == "tournament":
            self.pop[s].remove(self.__neg_tournament(s))
            self.pop[s].append(deepcopy(child))
        else:
            worst = self.get_worst(s)
            if worst.fitness < child.fitness:
                self.pop[s].remove(worst)
                self.pop[s].append(deepcopy(child))
        self.__update_archive(s, child)

    def __tournament(self, s: int) -> Ind:
        """Returns a parent selected by tournament."""
        tournament = random.sample(self.pop[s], Cons.T_SIZE)
        return max(tournament, key=attrgetter("fitness"))

    def __neg_tournament(self, s: int) -> Ind:
        """Returns an individual selected by negative tournament."""
        tournament = random.sample(self.pop[s], Cons.T_SIZE)
        return min(tournament, key=attrgetter("fitness"))

    def get_worst(self, s: int) -> Ind:
        """Returns the index of the least fit individual in the population."""
        return min(self.pop[s], key=attrgetter("fitness"))

    def get_best(self, s: int) -> Ind:
        """Returns the index of the best individual in the population."""
        return max(self.pop[s], key=attrgetter("fitness"))

    def get_best_fit(self, s: int) -> float:
        """Returns the fitness of the best individual in a given species."""
        return self.get_best(s).fitness

    def get_avg_fit(self, s: int) -> float:
        """Returns the average fitness of a given species."""
        return np.mean([p.fitness for p in self.pop[s]])

    def print_archive(self, s: int) -> None:
        """Prints the archived individuals."""
        for p in self.archive[s]:
            print(p.to_string())

    def print_pop(self) -> None:
        """Prints the current populations."""
        for s in range(Cons.S):
            print(f"Species {s}")
            for p in self.pop[s]:
                print(p.to_string())

    def run_ea(
        self, nkcs: NKCS, evals: np.ndarray, pbest: np.ndarray, pavg: np.ndarray
    ) -> None:
        """Executes a standard EA - partnering with the best candidates."""
        while self.evals < Cons.MAX_EVALS:
            for s in range(Cons.S):
                parent1 = self.__tournament(s)
                parent2 = self.__tournament(s)
                child = self.__create_offspring(parent1, parent2)
                team = self.__get_team_best(s, child)
                self.__eval_team(nkcs, team)
                self.__add_offspring(s, child)
                self.update_perf(evals, pbest, pavg)

    def __create_candidates(self, s: int) -> np.ndarray:
        """Returns M offspring genomes generated by 2 parents."""
        p1: Ind = self.__tournament(s)
        p2: Ind = self.__tournament(s)
        return np.asarray(
            [self.__create_offspring(p1, p2).genome for _ in range(Cons.M)]
        )

    def run_sea(
        self, nkcs: NKCS, evals: np.ndarray, pbest: np.ndarray, pavg: np.ndarray
    ) -> None:
        """Executes a surrogate-assisted EA."""
        while self.evals < Cons.MAX_EVALS:
            for s in range(Cons.S):
                X_train = np.asarray([p.genome for p in self.archive[s]])
                y_train = np.asarray([p.fitness for p in self.archive[s]])
                X_predict = self.__create_candidates(s)
                model = surrogate.Model()
                model.fit(X_train, y_train)
                scores = model.score(X_predict)
                # evaluate single best candidate
                child = Ind()
                child.genome = np.copy(X_predict[np.argmax(scores)])
                team = self.__get_team_best(s, child)
                self.__eval_team(nkcs, team)
                self.__add_offspring(s, child)
                self.update_perf(evals, pbest, pavg)
