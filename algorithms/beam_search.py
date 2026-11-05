"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

import random
from typing import List, Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)
        for name, value, minimum in (("beam_width", self.beam_width, 1),
                                     ("max_iterations", self.max_iterations, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.current_history = []
        self.beam_history = []
        self.path_diversity_history = []

    def _initial_beam(self) -> List[State]:
        """Seed unique configurations with randomized routes and departure slots.

        Sampling is bounded; if duplicates leave empty slots, expand discovered
        neighborhoods until K states exist or no additional state is reachable.
        A small state space can legitimately produce fewer than K configurations.
        """
        baseline = self.problem.generate_initial_state()
        beam = [baseline]
        seen = {baseline}
        times = self.problem.get_valid_time_domain()
        for _ in range(self.beam_width * 10):
            if len(beam) >= self.beam_width:
                break
            previous = {self.problem.origin: None}
            stack = [self.problem.origin]
            while stack:
                node = stack.pop()
                if node == self.problem.destination:
                    path = []
                    while node is not None:
                        path.append(node)
                        node = previous[node]
                    state = State(tuple(reversed(path)), random.choice(times))
                    if state not in seen:
                        beam.append(state)
                        seen.add(state)
                    break
                neighbors = self.problem.network.get_neighbors(node)
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor not in previous:
                        previous[neighbor] = node
                        stack.append(neighbor)

        # Appending during traversal supplies a bounded-by-K frontier without
        # repeatedly sampling duplicates in a small configuration space.
        for state in beam:
            if len(beam) >= self.beam_width:
                break
            for neighbor in get_neighbors(state, self.problem):
                if neighbor not in seen:
                    beam.append(neighbor)
                    seen.add(neighbor)
                    if len(beam) >= self.beam_width:
                        break
        return beam

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Guidelines:
        - Maintain a set of `beam_width` parallel configurations.
        - In each iteration, generate neighbors from all active states in the beam.
        - Pool and rank all generated successors collectively, then retain the top `beam_width` states.
        - Halt when the beam ceases to improve or reaches max_iterations.
        - Record the best cost of the population at each step in self.history.

        Encouraged:
        - Ensuring diversity across states in the beam to prevent collapse into a single redundant path.

        Prohibited:
        - Treating this as K independent, non-communicating searches.
        """
        self.history = []
        self.current_history = []
        self.beam_history = []
        self.path_diversity_history = []
        beam = self._initial_beam()

        def rank(states):
            # Keep insertion order on cost ties, avoiding hash-order dependence.
            unique = dict.fromkeys(states)
            return sorted(((state, self.problem.evaluate(state)) for state in unique),
                          key=lambda item: item[1])[:self.beam_width]

        ranked = rank(beam)
        best_state, best_cost = ranked[0]

        def record():
            self.history.append(best_cost)
            self.current_history.append(ranked[0][1])
            self.beam_history.append(tuple(state for state, _ in ranked))
            self.path_diversity_history.append(len({state.path for state, _ in ranked}))

        record()
        for _ in range(self.max_iterations):
            # Every parent contributes to one shared successor pool. Parents
            # themselves are not retained automatically (this is not elitist GA).
            candidates = [neighbor for state, _ in ranked
                          for neighbor in get_neighbors(state, self.problem)]
            if not candidates:
                break
            previous_costs = tuple(cost for _, cost in ranked)
            previous_costs += (float("inf"),) * (self.beam_width - len(ranked))
            ranked = rank(candidates)
            if ranked[0][1] < best_cost:
                best_state, best_cost = ranked[0]
            record()
            costs = tuple(cost for _, cost in ranked)
            costs += (float("inf"),) * (self.beam_width - len(ranked))
            # Compare the whole sorted cost vector: another beam can improve
            # while the current winner stays unchanged. Padding treats losing
            # a beam slot as deterioration rather than spurious improvement.
            if costs >= previous_costs:
                break

        return best_state, best_cost
