"""Operator contract tests; run with python -m unittest discover -s tests -v."""

from pathlib import Path
import random
import unittest

from algorithms.operators import crossover, get_neighbors
from core.congestion import GaussianPeakCongestion
from core.graph import RoadNetwork
from core.problem import CommuteProblem
from core.state import State


ROOT = Path(__file__).resolve().parents[1]


def make_problem(edges, t_min=420.0, t_max=435.0, t_step=5.0):
    network = RoadNetwork()
    for source, target in edges:
        network.add_node(source)
        network.add_node(target)
    for source, target in edges:
        network.add_edge(source, target, 1.0)
    return CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Uni",
                          t_min, t_max, t_step)


def simple_paths(problem):
    stack = [(problem.origin,)]
    while stack:
        path = stack.pop()
        if path[-1] == problem.destination:
            yield path
            continue
        for node in problem.network.get_neighbors(path[-1]):
            if node not in path:
                stack.append(path + (node,))


class OperatorTests(unittest.TestCase):
    def assert_valid(self, state, problem):
        self.assertTrue(problem.is_valid_state(state), state)
        self.assertIn(state.departure_time, problem.get_valid_time_domain())

    def test_time_neighbors_at_bounds_and_middle(self):
        problem = make_problem([("Home", "Uni")], t_max=432.0)
        for departure, expected in [(420, [425]), (425, [420, 430]), (430, [425])]:
            state = State(("Home", "Uni"), departure)
            neighbors = get_neighbors(state, problem)
            self.assertEqual([s.departure_time for s in neighbors], expected)
            self.assertNotIn(state, neighbors)

    def test_detours_shortcuts_dead_ends_and_back_edges(self):
        problem = make_problem([
            ("Home", "A"), ("A", "B"), ("B", "Uni"),
            ("A", "X"), ("X", "Y"), ("Y", "Z"), ("Z", "B"),
            ("Y", "Home"), ("A", "Dead"), ("Dead", "A"),
            ("Home", "B"), ("X", "X"),
        ])
        state = State(("Home", "A", "B", "Uni"), 425)
        neighbors = get_neighbors(state, problem)
        routes = {s.path for s in neighbors}
        self.assertIn(("Home", "A", "X", "Y", "Z", "B", "Uni"), routes)
        self.assertIn(("Home", "B", "Uni"), routes)
        self.assertEqual(len(neighbors), len(set(neighbors)))
        self.assertEqual(state, State(("Home", "A", "B", "Uni"), 425))
        for candidate in neighbors:
            self.assert_valid(candidate, problem)

    def test_route_change_when_only_one_departure_is_available(self):
        problem = make_problem([
            ("Home", "A"), ("A", "Uni"), ("Home", "Uni")
        ], t_max=421)
        neighbors = get_neighbors(State(("Home", "A", "Uni"), 420), problem)
        self.assertEqual(neighbors, [State(("Home", "Uni"), 420)])

    def test_isolated_state_has_no_neighbors(self):
        problem = make_problem([("Home", "Uni")], t_max=421)
        self.assertEqual(get_neighbors(State(("Home", "Uni"), 420), problem), [])

    def test_invalid_and_off_grid_states_are_rejected(self):
        problem = make_problem([("Home", "A"), ("A", "Uni"), ("A", "Home")])
        valid = State(("Home", "A", "Uni"), 420)
        for state in [State(valid.path, 419), State(valid.path, 421),
                      State(valid.path, float("nan")), State(("Home", "Uni"), 420),
                      State(("Home", "A", "Home", "A", "Uni"), 420)]:
            with self.assertRaises(ValueError):
                get_neighbors(state, problem)
            with self.assertRaises(ValueError):
                crossover(valid, state, problem)
            with self.assertRaises(ValueError):
                crossover(state, valid, problem)

    def test_crossover_at_shared_junction_and_time_rounding(self):
        problem = make_problem([
            ("Home", "A"), ("A", "J"), ("J", "B"), ("B", "Uni"),
            ("Home", "C"), ("C", "J"), ("J", "D"), ("D", "Uni"),
        ])
        parent1 = State(("Home", "A", "J", "B", "Uni"), 420)
        parent2 = State(("Home", "C", "J", "D", "Uni"), 435)
        for seed in range(20):
            random.seed(seed)
            child = crossover(parent1, parent2, problem)
            self.assert_valid(child, problem)
            self.assertEqual(child.departure_time, 425)
            self.assertIn(child.path, [
                ("Home", "A", "J", "D", "Uni"),
                ("Home", "C", "J", "B", "Uni"),
            ])

    def test_disjoint_parent_fallback_on_non_aligned_window(self):
        problem = make_problem([
            ("Home", "A"), ("A", "Uni"), ("Home", "B"), ("B", "Uni")
        ], t_max=432)
        parents = [State(("Home", "A", "Uni"), 425),
                   State(("Home", "B", "Uni"), 430)]
        for seed in range(10):
            random.seed(seed)
            child = crossover(*parents, problem)
            self.assert_valid(child, problem)
            self.assertIn(child.path, [parent.path for parent in parents])
            self.assertEqual(child.departure_time, 425)

    def test_crossover_rejects_cyclic_splices(self):
        problem = make_problem([
            ("Home", "A"), ("A", "B"), ("B", "Uni"),
            ("Home", "B"), ("B", "A"), ("A", "Uni"),
        ])
        parents = [State(("Home", "A", "B", "Uni"), 420),
                   State(("Home", "B", "A", "Uni"), 430)]
        for seed in range(20):
            random.seed(seed)
            child = crossover(*parents, problem)
            self.assert_valid(child, problem)
            self.assertIn(child.path, [("Home", "A", "Uni"), ("Home", "B", "Uni")])

    def test_identical_parents_and_random_seed_reproducibility(self):
        problem = make_problem([("Home", "Uni")])
        parent = State(("Home", "Uni"), 425)
        self.assertEqual(crossover(parent, parent, problem), parent)
        random.seed(42)
        first = crossover(parent, parent, problem)
        random.seed(42)
        self.assertEqual(crossover(parent, parent, problem), first)

    def test_zero_edge_route(self):
        network = RoadNetwork()
        network.add_node("Home")
        problem = CommuteProblem(network, GaussianPeakCongestion({}), "Home", "Home")
        parent = State(("Home",), 420)
        self.assertEqual(get_neighbors(parent, problem), [State(("Home",), 425)])
        self.assertEqual(crossover(parent, parent, problem), parent)

    def test_contracts_across_every_simple_route_in_bundled_maps(self):
        random.seed(42)
        for filename in ("toy_graph.json", "metropolis_graph.json", "downtown_grid.json"):
            problem = CommuteProblem(
                RoadNetwork.load_json(str(ROOT / "data" / "maps" / filename)),
                GaussianPeakCongestion({}), "Home", "Uni", 465, 525, 5,
            )
            paths = list(simple_paths(problem))
            self.assertTrue(paths)
            for path in paths:
                for departure in (465, 495, 525):
                    state = State(path, departure)
                    neighbors = get_neighbors(state, problem)
                    self.assertTrue(neighbors)
                    self.assertNotIn(state, neighbors)
                    self.assertEqual(len(neighbors), len(set(neighbors)))
                    for neighbor in neighbors:
                        self.assert_valid(neighbor, problem)
            for path1 in paths:
                for path2 in paths:
                    child = crossover(State(path1, 465), State(path2, 525), problem)
                    self.assert_valid(child, problem)


if __name__ == "__main__":
    unittest.main()
