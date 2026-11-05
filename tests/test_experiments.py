"""Benchmark accounting and CLI reproducibility checks."""

import contextlib
import io
import random
import unittest
from unittest.mock import patch

import main as runner
from core.congestion import GaussianPeakCongestion
from core.graph import RoadNetwork
from core.problem import CommuteProblem
from core.state import State
from experiments.benchmark import CountedProblem, exact_reference, summarize, trial, run


class ExperimentTests(unittest.TestCase):
    def problem(self):
        network = RoadNetwork()
        for node in ("Home", "A", "Uni"):
            network.add_node(node)
        network.add_edge("Home", "Uni", 3)
        network.add_edge("Home", "A", 1)
        network.add_edge("A", "Uni", 1)
        return CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Uni",
                              t_min=420, t_max=425, t_step=5)

    def test_exact_reference_enumerates_small_space_and_respects_cap(self):
        reference = exact_reference(self.problem())
        self.assertEqual(reference["cost"], 2)
        self.assertEqual(reference["paths"], 2)
        self.assertEqual(reference["states"], 4)
        with self.assertRaises(ValueError):
            exact_reference(self.problem(), max_states=3)

    def test_counting_tracks_improvements_and_all_objective_calls(self):
        problem = CountedProblem(self.problem())
        problem.evaluate(State(("Home", "Uni"), 420))
        problem.evaluate(State(("Home", "A", "Uni"), 420))
        problem.evaluate(State(("Home", "Uni"), 425))
        self.assertEqual(problem.evaluations, 3)
        self.assertEqual(problem.trace, [[1, 3], [2, 2]])

    def test_trial_resets_seed_and_preserves_accounting(self):
        problem = self.problem()
        reference = exact_reference(problem)
        params = {"max_restarts": 2, "max_sideways": 0}
        first, history = trial(problem, "hc", params, 42, reference)
        random.random()
        second, again = trial(problem, "hc", params, 42, reference)
        for key in first:
            if key != "runtime_s":
                self.assertEqual(first[key], second[key])
        self.assertEqual(history, again)
        self.assertEqual(first["cost"], 2)
        self.assertEqual(first["gap_pct"], 0)
        self.assertEqual(first["first_best_step"], 0)
        self.assertEqual(first["evaluations"], history["evaluation_trace"][-1][0])
        summary = summarize([first, second], ["algo"])[0]
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["std_cost"], 0)
        self.assertEqual(summary["optimal_rate"], 1)

    def test_tuning_seeds_cannot_overlap_evaluation_seeds(self):
        with self.assertRaises(ValueError):
            run({}, [42], [42], None)

    def test_cli_seed_is_independent_of_algorithm_order_and_saving_is_headless(self):
        state = State(("Home", "Uni"), 420)
        draws = []

        def execute(*args):
            draws.append(random.random())
            return state, 3, 0.01, [3]

        with patch("sys.argv", ["main.py", "--algo", "all", "--seed", "42",
                                 "--plot-route", "--plot-metrics", "--save-figs"]), \
             patch.object(runner, "load_configuration", return_value={}), \
             patch.object(runner, "build_problem", return_value=self.problem()), \
             patch.object(runner, "execute_solver", side_effect=execute), \
             patch.object(runner, "plot_commute_route") as routes, \
             patch.object(runner, "plot_convergence") as metrics, \
             contextlib.redirect_stdout(io.StringIO()):
            runner.main()
        self.assertEqual(len(draws), 4)
        self.assertEqual(len(set(draws)), 1)
        self.assertTrue(all(call.kwargs["show"] is False for call in routes.call_args_list))
        self.assertFalse(metrics.call_args.kwargs["show"])


if __name__ == "__main__":
    unittest.main()
