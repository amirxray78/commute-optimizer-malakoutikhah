"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

import math
import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)
        for name, value in (("max_sideways", self.max_sideways),
                            ("max_restarts", self.max_restarts)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        self.current_history = []
        self.restart_steps = []

    def _random_state(self) -> State:
        """Sample a DFS-tree route and a discrete departure, not uniformly over routes.

        Shuffle outgoing edges and visit each node once. Parent links reconstruct
        a continuous simple route even in networks with dead ends or cycles.
        Use the process-wide RNG so main.py's --seed controls restarts as well.
        """
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

    def solve(self) -> Tuple[State, float]:
        """
        Executes Hill Climbing optimization.

        Guidelines:
        - Search through the neighborhood greedily for candidates that minimize total travel time.
        - Handle flat local maxima / shoulders by allowing up to `max_sideways` non-improving steps.
        - Apply up to `max_restarts` to explore different starting basins.
        - Track the best objective cost seen at each major step in self.history.

        Prohibited:
        - Looping indefinitely on flat plateaus.
        """
        # history records the global incumbent; current_history also exposes
        # restart jumps. Both contain one entry per starting state/accepted move.
        self.history = []
        self.current_history = []
        self.restart_steps = []
        best_state = self.problem.generate_initial_state()
        best_cost = self.problem.evaluate(best_state)

        for restart in range(self.max_restarts + 1):
            current = best_state if restart == 0 else self._random_state()
            cost = self.problem.evaluate(current)
            sideways = 0
            self.restart_steps.append(len(self.history))
            if cost < best_cost:
                best_state, best_cost = current, cost
            self.history.append(best_cost)
            self.current_history.append(cost)

            while True:
                neighbors = get_neighbors(current, self.problem)
                if not neighbors:
                    break
                scored = [(candidate, self.problem.evaluate(candidate))
                          for candidate in neighbors]
                next_cost = min(value for _, value in scored)
                # A tiny absolute tolerance absorbs numerical noise in the FIFO
                # evaluator, while the sideways budget still bounds flat walks.
                if next_cost < cost - 1e-9:
                    sideways = 0
                elif math.isclose(next_cost, cost, rel_tol=0.0, abs_tol=1e-9):
                    if sideways >= self.max_sideways:
                        break
                    sideways += 1
                else:
                    break
                current = random.choice([candidate for candidate, value in scored
                                         if value == next_cost])
                cost = next_cost
                if cost < best_cost:
                    best_state, best_cost = current, cost
                self.history.append(best_cost)
                self.current_history.append(cost)

        return best_state, best_cost
