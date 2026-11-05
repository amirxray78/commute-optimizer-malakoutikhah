"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""

import math
import random
from typing import List, Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)
        self.tournament_size: int = self.params.get("tournament_size", 3)
        for name, value, minimum in (("population_size", self.population_size, 1),
                                     ("generations", self.generations, 0),
                                     ("elitism", self.elitism, 0),
                                     ("tournament_size", self.tournament_size, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.elitism > self.population_size:
            raise ValueError("elitism cannot exceed population_size.")
        for name, value in (("crossover_rate", self.crossover_rate),
                            ("mutation_rate", self.mutation_rate)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be a finite probability in [0, 1].")
        self.current_history = []
        self.population_history = []
        self.path_diversity_history = []

    def _random_state(self) -> State:
        """Sample a simple randomized DFS-tree route and a legal departure slot."""
        previous = {self.problem.origin: None}
        stack = [self.problem.origin]
        while stack:
            node = stack.pop()
            if node == self.problem.destination:
                path = []
                while node is not None:
                    path.append(node)
                    node = previous[node]
                return State(tuple(reversed(path)),
                             random.choice(self.problem.get_valid_time_domain()))
            neighbors = self.problem.network.get_neighbors(node)
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in previous:
                    previous[neighbor] = node
                    stack.append(neighbor)
        raise RuntimeError("No route connects the origin to the destination.")

    def _initial_population(self) -> List[State]:
        """Prefer unique initial samples, allowing duplicates in small state spaces.

        Bound retries so a single-state problem still terminates. Sampling DFS
        trees does not imply a uniform distribution over all feasible routes.
        """
        population = [self.problem.generate_initial_state()]
        seen = set(population)
        for _ in range(self.population_size * 10):
            if len(population) >= self.population_size:
                break
            candidate = self._random_state()
            if candidate not in seen:
                population.append(candidate)
                seen.add(candidate)
        while len(population) < self.population_size:
            population.append(random.choice(population))
        return population

    def _select_parent(self, scored: List[Tuple[State, float]]) -> State:
        """Tournament selection with fitness 1/(1+cost) for non-negative durations.

        Sample population positions, so repeated chromosomes retain their usual
        multiplicity. Clamp tournament size for very small populations.
        """
        contestants = random.sample(scored, min(self.tournament_size, len(scored)))
        return max(contestants, key=lambda item: 1.0 / (1.0 + item[1]))[0]

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Guidelines:
        - Initialize a population of valid candidate solutions.
        - Design a fitness metric suitable for minimizing commute travel time (lower cost -> higher fitness).
        - Use selection operators (such as tournament or roulette selection) to pick parents.
        - Pair parents to produce offspring via `crossover()`, and apply stochastic mutation.
        - Preserve the top candidates across generations if using elitism.
        - Track best population performance per generation in self.history.

        Encouraged:
        - Experimenting with population size, selection pressure, and operator rates to balance exploration vs. exploitation.
        """
        self.history = []
        self.current_history = []
        self.population_history = []
        self.path_diversity_history = []
        population = self._initial_population()
        best_state = None
        best_cost = float("inf")

        for generation in range(self.generations + 1):
            scored = sorted(((state, self.problem.evaluate(state)) for state in population),
                            key=lambda item: item[1])
            if best_state is None or scored[0][1] < best_cost:
                best_state, best_cost = scored[0]
            # Store both the archive and this generation's winner: with no
            # elitism, the current population can lose its previous best member.
            self.history.append(best_cost)
            self.current_history.append(scored[0][1])
            self.population_history.append(tuple(state for state, _ in scored))
            self.path_diversity_history.append(len({state.path for state, _ in scored}))
            if generation == self.generations:
                break

            offspring = [state for state, _ in scored[:self.elitism]]
            while len(offspring) < self.population_size:
                parent1 = self._select_parent(scored)
                if random.random() < self.crossover_rate:
                    parent2 = self._select_parent(scored)
                    child = crossover(parent1, parent2, self.problem)
                else:
                    child = parent1
                if random.random() < self.mutation_rate:
                    neighbors = get_neighbors(child, self.problem)
                    if neighbors:
                        child = random.choice(neighbors)
                offspring.append(child)
            population = offspring

        return best_state, best_cost
