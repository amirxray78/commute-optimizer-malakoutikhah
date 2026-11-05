"""Behavioral tests for genetic selection, variation, and elite preservation."""

import json
import math
from pathlib import Path
import random
import unittest
from unittest.mock import Mock, patch

from algorithms.genetic_algorithm import GeneticAlgorithmSolver
from core.congestion import GaussianPeakCongestion
from core.graph import RoadNetwork
from core.problem import CommuteProblem
from core.state import State


ROOT = Path(__file__).resolve().parents[1]


def landscape(costs):
    states = {name: State((name,), 420) for name in costs}
    problem = Mock()
    problem.evaluate.side_effect = lambda state: costs[state.path[0]]
    return problem, states


class GeneticAlgorithmTests(unittest.TestCase):
    def test_tournament_prefers_lower_travel_cost(self):
        problem, states = landscape({"slow": 20, "fast": 1, "middle": 10})
        solver = GeneticAlgorithmSolver(problem, tournament_size=10)
        scored = [(states[name], cost) for name, cost in
                  (("slow", 20), ("fast", 1), ("middle", 10))]
        self.assertEqual(solver._select_parent(scored), states["fast"])

    def test_crossover_creates_offspring_and_preserves_odd_population_size(self):
        problem, states = landscape({"parent": 10, "child": 1})
        solver = GeneticAlgorithmSolver(problem, population_size=3, generations=1,
                                        elitism=0, crossover_rate=1, mutation_rate=0)
        with patch.object(solver, "_initial_population", return_value=[states["parent"]] * 3), \
             patch("algorithms.genetic_algorithm.crossover", return_value=states["child"]) as cross, \
             patch("algorithms.genetic_algorithm.get_neighbors") as mutate:
            self.assertEqual(solver.solve(), (states["child"], 1))
        self.assertEqual(cross.call_count, 3)
        mutate.assert_not_called()
        self.assertEqual(solver.population_history[-1], (states["child"],) * 3)

    def test_mutation_uses_a_valid_neighbor(self):
        problem, states = landscape({"parent": 10, "mutant": 1})
        solver = GeneticAlgorithmSolver(problem, population_size=3, generations=1,
                                        elitism=0, crossover_rate=0, mutation_rate=1)
        with patch.object(solver, "_initial_population", return_value=[states["parent"]] * 3), \
             patch("algorithms.genetic_algorithm.crossover") as cross, \
             patch("algorithms.genetic_algorithm.get_neighbors",
                   return_value=[states["mutant"]]) as mutate:
            self.assertEqual(solver.solve(), (states["mutant"], 1))
        cross.assert_not_called()
        self.assertEqual(mutate.call_count, 3)

    def test_zero_rates_clone_selected_parent_without_variation(self):
        problem, states = landscape({"parent": 10})
        solver = GeneticAlgorithmSolver(problem, population_size=3, generations=1,
                                        elitism=0, crossover_rate=0, mutation_rate=0)
        with patch.object(solver, "_initial_population", return_value=[states["parent"]] * 3), \
             patch("algorithms.genetic_algorithm.crossover") as cross, \
             patch("algorithms.genetic_algorithm.get_neighbors") as mutate:
            self.assertEqual(solver.solve(), (states["parent"], 10))
        cross.assert_not_called()
        mutate.assert_not_called()
        self.assertEqual(solver.history, [10, 10])

    def test_elites_survive_destructive_mutation_unchanged(self):
        problem, states = landscape({"best": 1, "second": 2, "worst": 10})
        solver = GeneticAlgorithmSolver(problem, population_size=5, generations=1,
                                        elitism=2, crossover_rate=0, mutation_rate=1)
        initial = [states["best"], states["second"]] + [states["worst"]] * 3
        with patch.object(solver, "_initial_population", return_value=initial), \
             patch("algorithms.genetic_algorithm.get_neighbors",
                   return_value=[states["worst"]]) as mutate:
            self.assertEqual(solver.solve(), (states["best"], 1))
        self.assertEqual(solver.population_history[-1], tuple(initial))
        self.assertEqual(mutate.call_count, 3)

    def test_archive_retains_best_when_no_elitism_loses_it(self):
        problem, states = landscape({"best": 1, "worst": 10})
        solver = GeneticAlgorithmSolver(problem, population_size=3, generations=1,
                                        elitism=0, crossover_rate=0, mutation_rate=1)
        with patch.object(solver, "_initial_population", return_value=[states["best"]] * 3), \
             patch("algorithms.genetic_algorithm.get_neighbors", return_value=[states["worst"]]):
            self.assertEqual(solver.solve(), (states["best"], 1))
        self.assertEqual(solver.history, [1, 1])
        self.assertEqual(solver.current_history, [1, 10])

    def test_full_elitism_skips_breeding(self):
        problem, states = landscape({"parent": 1})
        solver = GeneticAlgorithmSolver(problem, population_size=1, elitism=1, generations=2)
        with patch.object(solver, "_initial_population", return_value=[states["parent"]]), \
             patch.object(solver, "_select_parent") as selection:
            self.assertEqual(solver.solve(), (states["parent"], 1))
        selection.assert_not_called()
        self.assertEqual(solver.history, [1, 1, 1])

    def test_zero_generations_and_repeated_solve_reset_statistics(self):
        problem, states = landscape({"slow": 10, "fast": 1})
        solver = GeneticAlgorithmSolver(problem, population_size=2, generations=0)
        with patch.object(solver, "_initial_population",
                          return_value=[states["slow"], states["fast"]]), \
             patch.object(solver, "_select_parent") as selection:
            for _ in range(2):
                self.assertEqual(solver.solve(), (states["fast"], 1))
                self.assertEqual(solver.history, [1])
                self.assertEqual(solver.path_diversity_history, [2])
                self.assertEqual(len(solver.population_history), 1)
        selection.assert_not_called()

    def test_mutation_with_empty_neighborhood_keeps_parent(self):
        problem, states = landscape({"parent": 1})
        solver = GeneticAlgorithmSolver(problem, population_size=1, elitism=0, generations=2,
                                        crossover_rate=0, mutation_rate=1)
        with patch.object(solver, "_initial_population", return_value=[states["parent"]]), \
             patch("algorithms.genetic_algorithm.get_neighbors", return_value=[]):
            self.assertEqual(solver.solve(), (states["parent"], 1))
        self.assertEqual(solver.history, [1, 1, 1])

    def test_invalid_parameters_are_rejected(self):
        for name in ("population_size", "tournament_size"):
            for value in (0, -1, 2.5, True, "2"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    GeneticAlgorithmSolver(Mock(), **{name: value})
        for name in ("generations", "elitism"):
            for value in (-1, 2.5, True, "2"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    GeneticAlgorithmSolver(Mock(), **{name: value})
        with self.assertRaises(ValueError):
            GeneticAlgorithmSolver(Mock(), population_size=2, elitism=3)
        for name in ("crossover_rate", "mutation_rate"):
            for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    GeneticAlgorithmSolver(Mock(), **{name: value})

    def test_initial_population_is_valid_diverse_and_fills_small_spaces(self):
        network = RoadNetwork()
        for node in ("Home", "A", "B", "Dead", "Uni"):
            network.add_node(node)
        for source, target in [("Home", "A"), ("A", "Uni"), ("Home", "B"),
                               ("B", "Uni"), ("B", "Home"), ("Home", "Dead")]:
            network.add_edge(source, target, 1)
        problem = CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Uni",
                                 t_min=420, t_max=430, t_step=5)
        solver = GeneticAlgorithmSolver(problem, population_size=20)
        random.seed(42)
        population = solver._initial_population()
        self.assertEqual(len(population), 20)
        self.assertEqual(len(set(population)), 6)
        self.assertEqual(population[0], problem.generate_initial_state())
        for state in population:
            self.assertTrue(problem.is_valid_state(state))
            self.assertIn(state.departure_time, problem.get_valid_time_domain())

    def test_single_state_space_terminates_without_duplicate_retry_loop(self):
        network = RoadNetwork()
        network.add_node("Home")
        problem = CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Home",
                                 t_min=420, t_max=421, t_step=5)
        solver = GeneticAlgorithmSolver(problem, population_size=5, generations=3,
                                        crossover_rate=1, mutation_rate=1)
        self.assertEqual(solver.solve(), (State(("Home",), 420), 0))
        self.assertEqual(solver.history, [0] * 4)
        self.assertTrue(all(len(population) == 5 for population in solver.population_history))

    def test_disconnected_network_fails_clearly(self):
        network = RoadNetwork()
        network.add_node("Home")
        network.add_node("Uni")
        problem = CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Uni")
        solver = GeneticAlgorithmSolver(problem)
        with self.assertRaises(RuntimeError):
            solver.solve()
        with self.assertRaises(RuntimeError):
            solver._random_state()

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
                solver = GeneticAlgorithmSolver(problem, **config["genetic_algorithm"])
                random.seed(42)
                state, cost = solver.solve()
                histories = (solver.history[:], solver.current_history[:],
                             solver.population_history[:], solver.path_diversity_history[:])
                self.assertTrue(problem.is_valid_state(state))
                self.assertTrue(math.isfinite(cost))
                self.assertAlmostEqual(cost, problem.evaluate(state))
                self.assertLessEqual(cost, problem.evaluate(problem.generate_initial_state()))
                self.assertEqual(cost, solver.history[-1])
                self.assertTrue(all(a >= b for a, b in zip(solver.history, solver.history[1:])))
                self.assertTrue(all(a >= b for a, b in
                                    zip(solver.current_history, solver.current_history[1:])))
                self.assertEqual(len(solver.history), solver.generations + 1)
                self.assertEqual(len(solver.history), len(solver.population_history))
                self.assertEqual(len(solver.history), len(solver.current_history))
                for population, diversity in zip(solver.population_history,
                                                 solver.path_diversity_history):
                    self.assertEqual(len(population), solver.population_size)
                    self.assertEqual(diversity, len({s.path for s in population}))
                    for candidate in population:
                        self.assertTrue(problem.is_valid_state(candidate))
                        self.assertIn(candidate.departure_time, problem.get_valid_time_domain())
                random.seed(42)
                self.assertEqual(solver.solve(), (state, cost))
                self.assertEqual((solver.history, solver.current_history, solver.population_history,
                                  solver.path_diversity_history), histories)


if __name__ == "__main__":
    unittest.main()
