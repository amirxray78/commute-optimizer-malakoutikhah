"""Behavioral tests for pooled selection, diversity, and beam termination."""

import json
import math
from pathlib import Path
import random
import unittest
from unittest.mock import Mock, patch

from algorithms.beam_search import LocalBeamSearchSolver
from core.congestion import GaussianPeakCongestion
from core.graph import RoadNetwork
from core.problem import CommuteProblem
from core.state import State


ROOT = Path(__file__).resolve().parents[1]


def landscape(costs, edges):
    states = {name: State((name,), 420) for name in costs}
    problem = Mock()
    problem.evaluate.side_effect = lambda state: costs[state.path[0]]
    neighbors = lambda state, _: [states[name] for name in edges.get(state.path[0], [])]
    return problem, states, neighbors


def small_problem(t_max=430):
    network = RoadNetwork()
    for node in ("Home", "A", "B", "Dead", "Uni"):
        network.add_node(node)
    for source, target in [("Home", "A"), ("A", "Uni"), ("Home", "B"),
                           ("B", "Uni"), ("B", "Home"), ("Home", "Dead")]:
        network.add_edge(source, target, 1)
    return CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Uni",
                          t_min=420, t_max=t_max, t_step=5)


class LocalBeamSearchTests(unittest.TestCase):
    def run_landscape(self, costs, edges, initial, **params):
        problem, states, neighbors = landscape(costs, edges)
        solver = LocalBeamSearchSolver(problem, **params)
        with patch.object(solver, "_initial_beam", return_value=[states[x] for x in initial]), \
             patch("algorithms.beam_search.get_neighbors", side_effect=neighbors):
            result = solver.solve()
        return solver, result, states

    def test_global_pool_can_select_multiple_successors_from_one_parent(self):
        solver, result, states = self.run_landscape(
            {"a": 10, "b": 20, "x": 1, "y": 2, "z": 3},
            {"a": ["x", "y"], "b": ["z"]}, ["a", "b"],
            beam_width=2, max_iterations=1,
        )
        self.assertEqual(result, (states["x"], 1))
        self.assertEqual(solver.beam_history[-1], (states["x"], states["y"]))

    def test_duplicate_successors_do_not_consume_beam_slots(self):
        solver, _, states = self.run_landscape(
            {"a": 10, "b": 20, "x": 1, "y": 2},
            {"a": ["x", "x"], "b": ["x", "y"]}, ["a", "b"],
            beam_width=2, max_iterations=1,
        )
        self.assertEqual(solver.beam_history[-1], (states["x"], states["y"]))
        self.assertEqual(solver.path_diversity_history, [2, 2])

    def test_non_winning_beam_improvement_allows_further_search(self):
        solver, result, states = self.run_landscape(
            {"a": 1, "b": 10, "c": 1, "d": 5, "goal": 0},
            {"a": ["c"], "b": ["d"], "d": ["goal"]}, ["a", "b"],
            beam_width=2,
        )
        self.assertEqual(result, (states["goal"], 0))
        self.assertEqual(solver.history, [1, 1, 0])

    def test_worse_successor_beam_preserves_best_return_value(self):
        solver, result, states = self.run_landscape(
            {"a": 1, "b": 2, "worse": 9}, {"a": ["worse"]}, ["a", "b"],
            beam_width=2,
        )
        self.assertEqual(result, (states["a"], 1))
        self.assertEqual(solver.history, [1, 1])
        self.assertEqual(solver.current_history, [1, 9])

    def test_equal_cost_cycle_stops(self):
        solver, _, _ = self.run_landscape(
            {"a": 1, "b": 1}, {"a": ["b"], "b": ["a"]}, ["a"],
            beam_width=1, max_iterations=100,
        )
        self.assertEqual(len(solver.history), 2)

    def test_shrinking_beam_is_not_mistaken_for_improvement(self):
        solver, _, _ = self.run_landscape(
            {"a": 1, "b": 2, "c": 1, "goal": 0},
            {"a": ["c"], "c": ["goal"]}, ["a", "b"], beam_width=2,
        )
        self.assertEqual(solver.history, [1, 1])
        self.assertEqual([len(beam) for beam in solver.beam_history], [2, 1])

    def test_empty_neighborhood_and_zero_budget_return_initial_best(self):
        for iterations in (0, 10):
            solver, result, states = self.run_landscape(
                {"a": 10, "b": 3}, {}, ["a", "b"],
                beam_width=2, max_iterations=iterations,
            )
            self.assertEqual(result, (states["b"], 3))
            self.assertEqual(solver.history, [3])

    def test_iteration_limit_bounds_strictly_improving_search(self):
        solver, result, states = self.run_landscape(
            {"a": 3, "b": 2, "c": 1}, {"a": ["b"], "b": ["c"]}, ["a"],
            beam_width=1, max_iterations=1,
        )
        self.assertEqual(result, (states["b"], 2))
        self.assertEqual(solver.history, [3, 2])

    def test_invalid_parameters_are_rejected(self):
        for width in (0, -1, 2.5, True, "2"):
            with self.assertRaises(ValueError):
                LocalBeamSearchSolver(Mock(), beam_width=width)
        for iterations in (-1, 2.5, True, "2"):
            with self.assertRaises(ValueError):
                LocalBeamSearchSolver(Mock(), max_iterations=iterations)

    def test_initialization_fills_small_state_space_without_duplicates(self):
        problem = small_problem()
        solver = LocalBeamSearchSolver(problem, beam_width=10)
        random.seed(42)
        beam = solver._initial_beam()
        self.assertEqual(len(beam), 6)  # Two routes, three departure slots.
        self.assertEqual(len(beam), len(set(beam)))
        self.assertEqual({s.path for s in beam},
                         {("Home", "A", "Uni"), ("Home", "B", "Uni")})
        self.assertEqual({s.departure_time for s in beam}, {420, 425, 430})
        for state in beam:
            self.assertTrue(problem.is_valid_state(state))

    def test_duplicate_sampling_falls_back_to_neighborhood_expansion(self):
        problem = small_problem()
        solver = LocalBeamSearchSolver(problem, beam_width=3)
        with patch("algorithms.beam_search.random.shuffle"), \
             patch("algorithms.beam_search.random.choice", return_value=420):
            beam = solver._initial_beam()
        self.assertEqual(len(beam), 3)
        self.assertEqual(len(set(beam)), 3)

    def test_single_state_space_terminates(self):
        network = RoadNetwork()
        network.add_node("Home")
        problem = CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Home",
                                 t_min=420, t_max=421, t_step=5)
        solver = LocalBeamSearchSolver(problem, beam_width=5)
        self.assertEqual(solver.solve(), (State(("Home",), 420), 0))
        self.assertEqual(solver.history, [0])

    def test_disconnected_network_fails_clearly(self):
        network = RoadNetwork()
        network.add_node("Home")
        network.add_node("Uni")
        problem = CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Uni")
        with self.assertRaises(RuntimeError):
            LocalBeamSearchSolver(problem).solve()

    def test_bundled_scenarios_are_valid_and_seed_reproducible(self):
        config = json.loads((ROOT / "config.json").read_text())
        for map_name, scenario in [
            ("toy_graph", "rush_hour"),
            ("metropolis_graph", "dual_peak_rush_hour"),
            ("metropolis_graph", "cascading_gridlock"),
            ("metropolis_graph", "crossover_showcase"),
            ("downtown_grid", "flat_plateau"),
        ]:
            with self.subTest(scenario=scenario):
                problem = CommuteProblem(
                    RoadNetwork.load_json(str(ROOT / "data" / "maps" / f"{map_name}.json")),
                    GaussianPeakCongestion.load_json(
                        str(ROOT / "data" / "scenarios" / f"{scenario}.json")),
                    **config["problem"],
                )
                solver = LocalBeamSearchSolver(problem, **config["beam_search"])
                random.seed(42)
                state, cost = solver.solve()
                histories = (solver.history[:], solver.current_history[:],
                             solver.beam_history[:], solver.path_diversity_history[:])
                self.assertTrue(problem.is_valid_state(state))
                self.assertTrue(math.isfinite(cost))
                self.assertAlmostEqual(cost, problem.evaluate(state))
                self.assertLessEqual(cost, problem.evaluate(problem.generate_initial_state()))
                self.assertEqual(cost, solver.history[-1])
                self.assertTrue(all(a >= b for a, b in zip(solver.history, solver.history[1:])))
                self.assertLessEqual(len(solver.history), solver.max_iterations + 1)
                self.assertEqual(len(solver.history), len(solver.beam_history))
                self.assertEqual(len(solver.history), len(solver.current_history))
                for beam, diversity in zip(solver.beam_history, solver.path_diversity_history):
                    self.assertLessEqual(len(beam), solver.beam_width)
                    self.assertEqual(len(beam), len(set(beam)))
                    self.assertEqual(diversity, len({s.path for s in beam}))
                    for candidate in beam:
                        self.assertTrue(problem.is_valid_state(candidate))
                        self.assertIn(candidate.departure_time, problem.get_valid_time_domain())
                random.seed(42)
                self.assertEqual(solver.solve(), (state, cost))
                self.assertEqual((solver.history, solver.current_history, solver.beam_history,
                                  solver.path_diversity_history), histories)


if __name__ == "__main__":
    unittest.main()
