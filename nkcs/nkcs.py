#!/usr/bin/python3
#
# Copyright (C) 2019--2024 Richard Preen <rpreen@gmail.com>
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

"""An implementation of the NKCS model for exploring aspects of coevolution."""

from __future__ import annotations

import logging
from typing import Final

import numpy as np

from .constants import Constants as Cons

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nkcs")


class NKCS:
    """NKCS model."""

    def __init__(self) -> None:
        """Initialise a randomly generated NKCS model."""
        self.species: list[NKCS.Species] = [
            self.Species(i) for i in range(Cons.S)
        ]

    def __calc_fit(self, sp: int, team: list[np.ndarray]) -> float:
        """Return the fitness of an individual partnered with a given team."""
        total: float = 0
        for i in range(Cons.N):
            inputs = self.__get_gene_inputs(sp, team, i)
            total += self.species[sp].gene_fit(inputs, i)
        return total / Cons.N

    def calc_team_fit(self, team: list[np.ndarray]) -> float:
        """Return the total team fitness."""
        total: float = 0
        for s in range(Cons.S):
            total += self.__calc_fit(s, team)
        return total

    def __get_inputs_internal(
        self, sp: int, team: list[np.ndarray], gene_idx: int
    ) -> list[float]:
        """Return gene internal inputs."""
        species: NKCS.Species = self.species[sp]  # species containing the gene
        offset: int = gene_idx * (species.n_gene_inputs - 1)  # map offset
        inputs: list[float] = []
        for i in range(Cons.K):
            node = species.map[offset + i]
            inputs.append(team[sp][node])
        return inputs

    def __get_inputs_external_line(
        self, sp: int, team: list[np.ndarray], gene_idx: int
    ) -> list[float]:
        """Return gene external inputs with line topology."""
        species: NKCS.Species = self.species[sp]  # species containing the gene
        offset: int = gene_idx * (species.n_gene_inputs - 1)  # map offset
        cnt: int = Cons.K
        inputs: list[float] = []
        if sp != 0:
            left = Cons.S - 1 if sp - 1 < 0 else sp - 1
            for _ in range(Cons.C):
                node = species.map[offset + cnt]
                inputs.append(team[left][node])
                cnt += 1
        if sp != Cons.S - 1:
            right = (sp + 1) % Cons.S
            for _ in range(Cons.C):
                node = species.map[offset + cnt]
                inputs.append(team[right][node])
                cnt += 1
        return inputs

    def __get_inputs_external_std(
        self, sp: int, team: list[np.ndarray], gene_idx: int
    ) -> list[float]:
        """Return gene external inputs with standard topology."""
        species: NKCS.Species = self.species[sp]  # species containing the gene
        offset: int = gene_idx * (species.n_gene_inputs - 1)  # map offset
        cnt: int = Cons.K
        inputs: list[float] = []
        for j in range(Cons.S):
            if j != sp:
                for _ in range(Cons.C):
                    node = species.map[offset + cnt]
                    inputs.append(team[j][node])
                    cnt += 1
        return inputs

    def __get_gene_inputs(
        self, sp: int, team: list[np.ndarray], gene_idx: int
    ) -> np.ndarray:
        """Return the inputs to a gene (including the internal state)."""
        # get internal connections
        inputs_int = self.__get_inputs_internal(sp, team, gene_idx)
        # get external connections
        if Cons.NKCS_TOPOLOGY == "line":
            inputs_ext = self.__get_inputs_external_line(sp, team, gene_idx)
        elif Cons.NKCS_TOPOLOGY == "standard":
            inputs_ext = self.__get_inputs_external_std(sp, team, gene_idx)
        else:
            raise ValueError("unsupported NKCS topology")
        # add internal state
        inputs = inputs_int + inputs_ext + team[sp][gene_idx]
        return np.array(inputs)

    def display(self, sp: int) -> None:
        """Print a specified NKCS species."""
        logger.info("**********************")
        logger.info("{sp} SPECIES:")
        logger.info("**********************")
        self.species[sp].display()

    class Species:
        """A species within an NKCS model."""

        def __init__(self, sp: int) -> None:
            """Initialise a species with random connectivity."""
            x: int = 0  # number of coevolving species
            if Cons.S > 1:
                if Cons.NKCS_TOPOLOGY == "line":
                    x = 1 if sp in (0, Cons.S - 1) else 2
                elif Cons.NKCS_TOPOLOGY == "standard":
                    x = Cons.S - 1
                else:
                    raise ValueError("unsupported NKCS topology")
            # n inputs to each gene
            self.n_gene_inputs: Final[int] = Cons.K + (x * Cons.C) + 1
            # connectivity length
            map_length: Final[int] = Cons.N * self.n_gene_inputs - 1
            # connectivity
            self.map: np.ndarray = np.random.randint(0, Cons.N, map_length)
            # each gene's hash table
            self.ftable: list[dict[tuple, float]] = [{} for i in range(Cons.N)]

        def gene_fit(self, inputs: np.ndarray, gene: int) -> float:
            """Return the fitness of an individual gene within a species."""
            # find fitness in table
            key = tuple(inputs)
            fit = self.ftable[gene].get(key)
            if fit is None:  # not found, add new
                fitness = np.random.uniform(low=0, high=1)
                self.ftable[gene][key] = fitness
            else:
                fitness = fit
            return fitness

        def display(self) -> None:
            """Print an NKCS species."""
            logger.info(f"con: {self.map}")
            logger.info("fitness table:")
            for i, fitness in enumerate(self.ftable):
                logger.info(f"Gene {i}")
                logger.info(fitness)
