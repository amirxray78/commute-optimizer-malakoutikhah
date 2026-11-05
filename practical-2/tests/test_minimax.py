"""Minimax decisions, deadline behavior, and isolation from the live game."""

import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import Action, GhostBehavior
import student_assignment as agents
from test_evaluation import scenario


class TreeGame:
    """Known tree: depth one prefers RIGHT; minimax depth two prefers LEFT."""

    edges = {
        "root": {Action.LEFT: "left", Action.RIGHT: "right"},
        "left": {Action.UP: "l3", Action.DOWN: "l5"},
        "right": {Action.UP: "r2", Action.DOWN: "r9"},
    }
    values = {"root": 0, "left": 0, "right": 10, "l3": 3, "l5": 5, "r2": 2, "r9": 9}

    def __init__(self, node="root", done=False):
        self.node, self.done = node, done

    def get_legal_actions(self):
        return list(self.edges.get(self.node, {}))

    def clone(self):
        return TreeGame(self.node, self.done)

    def step(self, action):
        self.node = self.edges[self.node][action]


class MinimaxTests(unittest.TestCase):
    def test_root_choice_reflects_minimizing_reply(self):
        with patch.object(agents, "evaluate_state", side_effect=lambda g: g.values[g.node]):
            self.assertEqual(agents.minimax_agent({}, TreeGame(), depth=1), Action.RIGHT)
            self.assertEqual(agents.minimax_agent({}, TreeGame(), depth=2), Action.LEFT)

    def test_recursive_max_and_min_levels_explore_all_leaves(self):
        with patch.object(agents, "evaluate_state", side_effect=lambda g: g.values[g.node]) as evaluate:
            self.assertEqual(agents._minimax(TreeGame(), 2, True), 3)
            self.assertEqual(evaluate.call_count, 4)
            evaluate.reset_mock()
            self.assertEqual(agents._minimax(TreeGame(), 2, False), 5)
            self.assertEqual(evaluate.call_count, 4)

    def test_terminal_cutoff_and_empty_actions_use_evaluation(self):
        with patch.object(agents, "evaluate_state", return_value=17):
            self.assertEqual(agents._minimax(TreeGame(), 0, True), 17)
            self.assertEqual(agents._minimax(TreeGame(), -1, True), 17)
            self.assertEqual(agents._minimax(TreeGame(done=True), 5, False), 17)
            self.assertEqual(agents._minimax(TreeGame("l3"), 5, True), 17)
        self.assertEqual(agents.minimax_agent({}, TreeGame(done=True)), Action.STAY)
        self.assertEqual(agents.minimax_agent({}, TreeGame("l3")), Action.STAY)

    def test_ties_keep_first_legal_action(self):
        with patch.object(agents, "evaluate_state", return_value=1):
            self.assertEqual(agents.minimax_agent({}, TreeGame(), depth=2), Action.LEFT)

    def test_invalid_depths_are_rejected(self):
        for depth in (0, -1, 1.5, True, "3"):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                agents.minimax_agent({}, TreeGame(), depth=depth)

    def test_timeout_before_first_depth_returns_legal_fallback(self):
        with patch.object(agents, "_check_search_deadline", side_effect=agents._SearchTimeout):
            self.assertEqual(agents.minimax_agent({}, TreeGame()), Action.LEFT)
        self.assertIsNone(agents._SEARCH_DEADLINE.get())

    def test_incomplete_depth_does_not_override_last_completed_result(self):
        def score(game, depth, maximising):
            if depth > 0:
                raise agents._SearchTimeout()
            return game.values[game.node]

        with patch.object(agents, "_minimax", side_effect=score):
            self.assertEqual(agents.minimax_agent({}, TreeGame(), depth=3), Action.RIGHT)
        self.assertIsNone(agents._SEARCH_DEADLINE.get())

    def test_deadline_clock_is_checked(self):
        token = agents._SEARCH_DEADLINE.set(10.0)
        try:
            with patch.object(agents.time, "perf_counter", return_value=10.0):
                with self.assertRaises(agents._SearchTimeout):
                    agents._minimax(TreeGame(), 2, True)
        finally:
            agents._SEARCH_DEADLINE.reset(token)

    def test_successor_restores_rng_when_search_is_interrupted(self):
        game = scenario(ammo=1)
        before, rng = game.get_state(), random.getstate()
        with self.assertRaises(agents._SearchTimeout):
            with agents._search_successor(game, game.get_legal_actions()[0]):
                random.random()
                raise agents._SearchTimeout()
        self.assertEqual(rng, random.getstate())
        self.assertEqual(before, game.get_state())

    def test_live_state_and_rng_remain_unchanged_and_result_repeats(self):
        game = scenario(ghosts=[(5, 3, GhostBehavior.DFS, True)], ammo=1)
        before, rng = game.get_state(), random.getstate()
        action = agents.minimax_agent({}, game, depth=2)
        self.assertIn(action, game.get_legal_actions())
        self.assertEqual(before, game.get_state())
        self.assertEqual(rng, random.getstate())
        self.assertEqual(action, agents.minimax_agent({}, game, depth=2))

    def test_real_game_prefers_killing_adjacent_final_ghost(self):
        game = scenario(ghosts=[(4, 3, GhostBehavior.FOLLOW, True)], ammo=1)
        action = agents.minimax_agent({}, game, depth=2)
        self.assertIn(action, game.get_legal_actions())
        child = game.clone()
        child.step(action)
        self.assertTrue(child.player.alive)
        self.assertFalse(any(ghost.alive for ghost in child.ghosts))

    def test_real_game_avoids_staying_next_to_ghost_without_ammo(self):
        game = scenario(ghosts=[(4, 3, GhostBehavior.FOLLOW, True)], ammo=0)
        action = agents.minimax_agent({}, game, depth=1)
        self.assertNotEqual(action, Action.STAY)
        child = game.clone()
        child.step(action)
        self.assertTrue(child.player.alive)


if __name__ == "__main__":
    unittest.main()
