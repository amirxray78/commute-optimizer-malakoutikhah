"""Behavioral tests for Metropolis acceptance, cooling, and best-state retention."""

import json
import math
from pathlib import Path
import random
import unittest
from unittest.mock import Mock, patch

from algorithms.simulated_annealing import SimulatedAnnealingSolver
from core.congestion import GaussianPeakCongestion
from core.graph import RoadNetwork
from core.problem import CommuteProblem
from core.state import State


ROOT = Path(__file__).resolve().parents[1]


def landscape(costs, edges):
    states = {name: State((name,), 420) for name in costs}
    problem = Mock()
    problem.generate_initial_state.return_value = states[next(iter(costs))]
    problem.evaluate.side_effect = lambda state: costs[state.path[0]]
    neighbors = lambda state, _: [states[name] for name in edges.get(state.path[0], [])]
    return problem, states, neighbors


class SimulatedAnnealingTests(unittest.TestCase):
    def test_improving_and_equal_cost_moves_are_always_accepted(self):
        problem, states, neighbors = landscape(
            {"start": 10, "equal": 10, "goal": 2},
            {"start": ["equal"], "equal": ["goal"]},
        )
        with patch("algorithms.simulated_annealing.get_neighbors", side_effect=neighbors), \
             patch("algorithms.simulated_annealing.random.random") as acceptance_draw:
            solver = SimulatedAnnealingSolver(problem)
            self.assertEqual(solver.solve(), (states["goal"], 2))
        acceptance_draw.assert_not_called()
        self.assertEqual(solver.current_history, [10, 10, 2])

    def test_metropolis_threshold_accepts_or_rejects_uphill_move(self):
        problem, states, neighbors = landscape(
            {"start": 1, "uphill": 3}, {"start": ["uphill"]},
        )
        # delta/T = 1, so acceptance probability is exp(-1), about 0.368.
        for draw, expected_current in ((0.2, 3), (0.8, 1)):
            with self.subTest(draw=draw), \
                 patch("algorithms.simulated_annealing.get_neighbors", side_effect=neighbors), \
                 patch("algorithms.simulated_annealing.random.random", return_value=draw):
                solver = SimulatedAnnealingSolver(problem, initial_temp=2, max_iterations=1)
                self.assertEqual(solver.solve(), (states["start"], 1))
                self.assertEqual(solver.current_history, [1, expected_current])
                self.assertEqual(solver.history, [1, 1])

    def test_uphill_escape_and_later_worse_move_preserve_best_state(self):
        problem, states, neighbors = landscape(
            {"start": 10, "barrier": 12, "goal": 1, "worse": 20},
            {"start": ["barrier"], "barrier": ["goal"], "goal": ["worse"]},
        )
        with patch("algorithms.simulated_annealing.get_neighbors", side_effect=neighbors), \
             patch("algorithms.simulated_annealing.random.random", return_value=0.0):
            solver = SimulatedAnnealingSolver(problem)
            self.assertEqual(solver.solve(), (states["goal"], 1))
        self.assertEqual(solver.current_history, [10, 12, 1, 20])
        self.assertEqual(solver.history, [10, 10, 1, 1])

    def test_cooling_changes_acceptance_for_same_energy_increase(self):
        problem, states, neighbors = landscape(
            {"start": 0, "middle": 0.5, "end": 1},
            {"start": ["middle"], "middle": ["end"]},
        )
        # The same draw accepts delta=0.5 at T=1, but rejects it at T=0.5.
        with patch("algorithms.simulated_annealing.get_neighbors", side_effect=neighbors), \
             patch("algorithms.simulated_annealing.random.random", return_value=0.5):
            solver = SimulatedAnnealingSolver(
                problem, initial_temp=1, cooling_rate=0.5, max_iterations=2,
            )
            self.assertEqual(solver.solve(), (states["start"], 0))
        self.assertEqual(solver.current_history, [0, 0.5, 0.5])
        self.assertEqual(solver.temperature_history, [1, 0.5, 0.25])

    def test_rejections_still_cool_and_stop_at_minimum_temperature(self):
        problem, _, neighbors = landscape(
            {"start": 1, "uphill": 100}, {"start": ["uphill"]},
        )
        with patch("algorithms.simulated_annealing.get_neighbors", side_effect=neighbors), \
             patch("algorithms.simulated_annealing.random.random", return_value=0.9):
            solver = SimulatedAnnealingSolver(
                problem, initial_temp=1, cooling_rate=0.5, min_temp=0.26,
            )
            solver.solve()
        self.assertEqual(solver.current_history, [1, 1, 1])
        self.assertEqual(solver.temperature_history, [1, 0.5, 0.25])

    def test_iteration_budget_bounds_flat_walk(self):
        problem, _, neighbors = landscape(
            {"start": 1, "other": 1}, {"start": ["other"], "other": ["start"]},
        )
        with patch("algorithms.simulated_annealing.get_neighbors", side_effect=neighbors):
            solver = SimulatedAnnealingSolver(problem, max_iterations=3)
            solver.solve()
        self.assertEqual(solver.history, [1] * 4)

    def test_zero_budget_and_initial_temperature_at_cutoff_do_not_propose(self):
        problem, states, _ = landscape({"start": 1}, {})
        for params in ({"max_iterations": 0}, {"initial_temp": 1, "min_temp": 1},
                       {"initial_temp": 1, "min_temp": 2}):
            with self.subTest(params=params), \
                 patch("algorithms.simulated_annealing.get_neighbors") as neighbors:
                solver = SimulatedAnnealingSolver(problem, **params)
                self.assertEqual(solver.solve(), (states["start"], 1))
                neighbors.assert_not_called()
                self.assertEqual(solver.history, [1])

    def test_empty_neighborhood_and_repeated_solve_reset_histories(self):
        problem, states, neighbors = landscape({"start": 1}, {})
        with patch("algorithms.simulated_annealing.get_neighbors", side_effect=neighbors):
            solver = SimulatedAnnealingSolver(problem)
            for _ in range(2):
                self.assertEqual(solver.solve(), (states["start"], 1))
                self.assertEqual(solver.history, [1])
                self.assertEqual(solver.current_history, [1])
                self.assertEqual(solver.temperature_history, [solver.initial_temp])

    def test_tiny_temperature_rejects_large_uphill_without_overflow(self):
        problem, states, neighbors = landscape(
            {"start": 1, "uphill": 2}, {"start": ["uphill"]},
        )
        with patch("algorithms.simulated_annealing.get_neighbors", side_effect=neighbors), \
             patch("algorithms.simulated_annealing.random.random", return_value=0.1):
            solver = SimulatedAnnealingSolver(
                problem, initial_temp=1e-300, min_temp=1e-310, max_iterations=1,
            )
            self.assertEqual(solver.solve(), (states["start"], 1))
        self.assertEqual(solver.current_history, [1, 1])

    def test_invalid_parameters_rejected(self):
        for name in ("initial_temp", "min_temp", "cooling_rate"):
            for value in (0, -1, float("inf"), float("nan"), True, "1"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    SimulatedAnnealingSolver(Mock(), **{name: value})
        for rate in (1, 2):
            with self.assertRaises(ValueError):
                SimulatedAnnealingSolver(Mock(), cooling_rate=rate)
        for iterations in (-1, 2.5, True, "3"):
            with self.assertRaises(ValueError):
                SimulatedAnnealingSolver(Mock(), max_iterations=iterations)

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
                solver = SimulatedAnnealingSolver(problem, **config["simulated_annealing"])
                random.seed(42)
                state, cost = solver.solve()
                histories = (solver.history[:], solver.current_history[:],
                             solver.temperature_history[:])
                self.assertTrue(problem.is_valid_state(state))
                self.assertIn(state.departure_time, problem.get_valid_time_domain())
                self.assertTrue(math.isfinite(cost))
                self.assertAlmostEqual(cost, problem.evaluate(state))
                self.assertLessEqual(cost, problem.evaluate(problem.generate_initial_state()))
                self.assertEqual(cost, solver.history[-1])
                self.assertTrue(all(a >= b for a, b in zip(solver.history, solver.history[1:])))
                self.assertEqual(len(solver.history), len(solver.current_history))
                self.assertEqual(len(solver.history), len(solver.temperature_history))
                self.assertLessEqual(len(solver.history), solver.max_iterations + 1)
                random.seed(42)
                self.assertEqual(solver.solve(), (state, cost))
                self.assertEqual((solver.history, solver.current_history,
                                  solver.temperature_history), histories)


if __name__ == "__main__":
    unittest.main()
