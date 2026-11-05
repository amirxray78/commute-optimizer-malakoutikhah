"""
algorithms/simulated_annealing.py
Stochastic optimization governed by a temperature schedule.
"""

import math
import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class SimulatedAnnealingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp: float = self.params.get("initial_temp", 100.0)
        self.cooling_rate: float = self.params.get("cooling_rate", 0.995)
        self.min_temp: float = self.params.get("min_temp", 1e-3)
        self.max_iterations: int = self.params.get("max_iterations", 2000)
        for name, value in (("initial_temp", self.initial_temp),
                            ("cooling_rate", self.cooling_rate),
                            ("min_temp", self.min_temp)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number.")
        if self.cooling_rate >= 1:
            raise ValueError("cooling_rate must be strictly between zero and one.")
        if (isinstance(self.max_iterations, bool)
                or not isinstance(self.max_iterations, int) or self.max_iterations < 0):
            raise ValueError("max_iterations must be a non-negative integer.")
        self.current_history = []
        self.temperature_history = []

    def solve(self) -> Tuple[State, float]:
        """
        Executes Simulated Annealing optimization.

        Guidelines:
        - Start from a high temperature where worse moves are accepted with significant probability.
        - Lower temperature over iterations according to a cooling schedule.
        - Downhill moves (improvements) must always be accepted.
        - Uphill moves (worse costs) should be accepted probabilistically according to temperature.
        - Record objective values throughout the run in self.history.

        Prohibited:
        - Ignoring the temperature parameter or turning this into pure hill climbing.
        """
        current = self.problem.generate_initial_state()
        current_cost = self.problem.evaluate(current)
        best_state, best_cost = current, current_cost
        temperature = self.initial_temp
        # Record the initial state, then every proposal, including rejections.
        # history is the incumbent; current_history exposes uphill excursions.
        # temperature_history holds the temperature for the next proposal.
        self.history = [best_cost]
        self.current_history = [current_cost]
        self.temperature_history = [temperature]

        for _ in range(self.max_iterations):
            if temperature <= self.min_temp:
                break
            neighbors = get_neighbors(current, self.problem)
            if not neighbors:
                break
            candidate = random.choice(neighbors)
            candidate_cost = self.problem.evaluate(candidate)
            delta = candidate_cost - current_cost
            if delta <= 0 or random.random() < math.exp(-delta / temperature):
                current, current_cost = candidate, candidate_cost
                if current_cost < best_cost:
                    best_state, best_cost = current, current_cost

            # Cool after every proposal, even if it was rejected. Otherwise a
            # rejected uphill move could hold the search at one temperature.
            temperature *= self.cooling_rate
            self.history.append(best_cost)
            self.current_history.append(current_cost)
            self.temperature_history.append(temperature)

        return best_state, best_cost
