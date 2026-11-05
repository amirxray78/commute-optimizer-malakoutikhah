"""Behavioral tests for steepest descent, plateaus, and random restarts."""

from pathlib import Path
import json
import math
import random
import unittest
from unittest.mock import Mock, patch

from algorithms.hill_climbing import HillClimbingSolver
from core.congestion import GaussianPeakCongestion
from core.graph import RoadNetwork
from core.problem import CommuteProblem
from core.state import State


ROOT = Path(__file__).resolve().parents[1]


def landscape(costs, edges):
    """Small explicit search landscape for testing algorithm decisions."""
    states = {name: State((name,), 420) for name in costs}
    problem = Mock()
    problem.generate_initial_state.return_value = states[next(iter(costs))]
    problem.evaluate.side_effect = lambda state: costs[state.path[0]]
    neighbors = lambda state, _: [states[name] for name in edges.get(state.path[0], [])]
    return problem, states, neighbors


class HillClimbingTests(unittest.TestCase):
    def test_selects_steepest_improvement_and_stops_before_uphill(self):
        problem, states, neighbors = landscape(
            {"start": 10, "small": 8, "large": 3, "uphill": 4},
            {"start": ["small", "large"], "large": ["uphill"]},
        )
        with patch("algorithms.hill_climbing.get_neighbors", side_effect=neighbors):
            solver = HillClimbingSolver(problem, max_restarts=0)
            self.assertEqual(solver.solve(), (states["large"], 3))
        self.assertEqual(solver.history, [10, 3])

    def test_sideways_move_crosses_shoulder(self):
        problem, states, neighbors = landscape(
            {"start": 10, "shoulder": 10, "goal": 2},
            {"start": ["shoulder"], "shoulder": ["goal"]},
        )
        with patch("algorithms.hill_climbing.get_neighbors", side_effect=neighbors):
            disabled = HillClimbingSolver(problem, max_restarts=0, max_sideways=0)
            self.assertEqual(disabled.solve(), (states["start"], 10))
            enabled = HillClimbingSolver(problem, max_restarts=0, max_sideways=1)
            self.assertEqual(enabled.solve(), (states["goal"], 2))
        self.assertEqual(enabled.history, [10, 10, 2])

    def test_flat_cycle_terminates_at_sideways_budget(self):
        problem, _, neighbors = landscape(
            {"start": 5, "other": 5}, {"start": ["other"], "other": ["start"]},
        )
        with patch("algorithms.hill_climbing.get_neighbors", side_effect=neighbors):
            solver = HillClimbingSolver(problem, max_restarts=0, max_sideways=3)
            self.assertEqual(solver.solve()[1], 5)
        self.assertEqual(solver.history, [5] * 4)

    def test_improvement_resets_consecutive_sideways_budget(self):
        problem, states, neighbors = landscape(
            {"start": 10, "a": 10, "b": 8, "c": 8, "goal": 1},
            {"start": ["a"], "a": ["b"], "b": ["c"], "c": ["goal"]},
        )
        with patch("algorithms.hill_climbing.get_neighbors", side_effect=neighbors):
            solver = HillClimbingSolver(problem, max_restarts=0, max_sideways=1)
            self.assertEqual(solver.solve(), (states["goal"], 1))
        self.assertEqual(solver.history, [10, 10, 8, 8, 1])

    def test_restarts_escape_basin_and_preserve_best_across_worse_restart(self):
        problem, states, neighbors = landscape(
            {"start": 10, "restart1": 12, "goal": 1, "restart2": 20},
            {"restart1": ["goal"]},
        )
        solver = HillClimbingSolver(problem, max_restarts=2)
        with patch("algorithms.hill_climbing.get_neighbors", side_effect=neighbors), \
             patch.object(solver, "_random_state", side_effect=[states["restart1"],
                                                               states["restart2"]]) as restart:
            self.assertEqual(solver.solve(), (states["goal"], 1))
            self.assertEqual(restart.call_count, 2)
        self.assertEqual(solver.current_history, [10, 12, 1, 20])
        self.assertEqual(solver.history, [10, 10, 1, 1])
        self.assertEqual(solver.restart_steps, [0, 1, 3])

    def test_solve_resets_histories(self):
        problem, _, neighbors = landscape({"start": 1}, {})
        solver = HillClimbingSolver(problem, max_restarts=0)
        with patch("algorithms.hill_climbing.get_neighbors", side_effect=neighbors):
            solver.solve()
            solver.solve()
        self.assertEqual(solver.history, [1])
        self.assertEqual(solver.current_history, [1])
        self.assertEqual(solver.restart_steps, [0])

    def test_invalid_budgets_rejected(self):
        for name in ("max_sideways", "max_restarts"):
            for value in (-1, 1.5, True, "2"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    HillClimbingSolver(Mock(), **{name: value})

    def test_random_restarts_are_valid_diverse_and_reproducible(self):
        network = RoadNetwork()
        for node in ("Home", "A", "B", "Dead", "Uni"):
            network.add_node(node)
        for source, target in [("Home", "A"), ("A", "Uni"), ("Home", "B"),
                               ("B", "Uni"), ("B", "Home"), ("Home", "Dead")]:
            network.add_edge(source, target, 1)
        problem = CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Uni")
        solver = HillClimbingSolver(problem)
        random.seed(42)
        first = [solver._random_state() for _ in range(50)]
        random.seed(42)
        self.assertEqual(first, [solver._random_state() for _ in range(50)])
        self.assertGreater(len({s.path for s in first}), 1)
        self.assertGreater(len({s.departure_time for s in first}), 1)
        for state in first:
            self.assertTrue(problem.is_valid_state(state))
            self.assertIn(state.departure_time, problem.get_valid_time_domain())

    def test_disconnected_graph_fails_clearly(self):
        network = RoadNetwork()
        network.add_node("Home")
        network.add_node("Uni")
        problem = CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Uni")
        solver = HillClimbingSolver(problem)
        with self.assertRaises(RuntimeError):
            solver.solve()
        with self.assertRaises(RuntimeError):
            solver._random_state()

    def test_bundled_scenarios_return_valid_incumbents_with_seeded_runs(self):
        config = json.loads((ROOT / "config.json").read_text())
        scenarios = [
            ("toy_graph", "rush_hour"),
            ("metropolis_graph", "dual_peak_rush_hour"),
            ("metropolis_graph", "cascading_gridlock"),
            ("metropolis_graph", "crossover_showcase"),
            ("downtown_grid", "flat_plateau"),
        ]
        for map_name, scenario in scenarios:
            with self.subTest(scenario=scenario):
                problem = CommuteProblem(
                    RoadNetwork.load_json(str(ROOT / "data" / "maps" / f"{map_name}.json")),
                    GaussianPeakCongestion.load_json(
                        str(ROOT / "data" / "scenarios" / f"{scenario}.json")),
                    **config["problem"],
                )
                solver = HillClimbingSolver(problem, **config["hill_climbing"])
                random.seed(42)
                state, cost = solver.solve()
                first_history = solver.history[:]
                self.assertTrue(problem.is_valid_state(state))
                self.assertIn(state.departure_time, problem.get_valid_time_domain())
                self.assertTrue(math.isfinite(cost))
                self.assertAlmostEqual(cost, problem.evaluate(state))
                self.assertLessEqual(cost, problem.evaluate(problem.generate_initial_state()))
                self.assertEqual(cost, solver.history[-1])
                self.assertTrue(all(a >= b for a, b in zip(solver.history, solver.history[1:])))
                self.assertEqual(len(solver.restart_steps), solver.max_restarts + 1)
                random.seed(42)
                self.assertEqual(solver.solve(), (state, cost))
                self.assertEqual(solver.history, first_history)


if __name__ == "__main__":
    unittest.main()
