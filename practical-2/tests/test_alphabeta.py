"""Alpha-beta equivalence with minimax, pruning, ordering, and isolation."""

import math
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import Action, GameConfig, GhostBehavior, PacmanGame
import student_assignment as agents
from test_evaluation import scenario
from test_minimax import TreeGame


def root_value(game, action, depth):
    """Minimax value of playing `action` at the root of a depth-`depth` search."""
    with agents._search_successor(game, action) as child:
        return agents._minimax(child, depth - 1, maximising=False)


class AlphaBetaTests(unittest.TestCase):
    def test_root_choice_matches_minimax_on_known_tree(self):
        with patch.object(agents, "evaluate_state", side_effect=lambda g: g.values[g.node]):
            self.assertEqual(agents.alphabeta_agent({}, TreeGame(), depth=1), Action.RIGHT)
            self.assertEqual(agents.alphabeta_agent({}, TreeGame(), depth=2), Action.LEFT)

    def test_pruning_skips_a_leaf_that_cannot_matter(self):
        with patch.object(agents, "evaluate_state", side_effect=lambda g: g.values[g.node]) as evaluate:
            value = agents._alphabeta(TreeGame(), 2, -math.inf, math.inf, True)
        self.assertEqual(value, 3)
        self.assertEqual(evaluate.call_count, 3)  # r9 is pruned after r2 <= 3

    def test_terminal_cutoff_and_empty_actions_use_evaluation(self):
        with patch.object(agents, "evaluate_state", return_value=17):
            self.assertEqual(agents._alphabeta(TreeGame(), 0, -math.inf, math.inf, True), 17)
            self.assertEqual(agents._alphabeta(TreeGame(done=True), 4, -math.inf, math.inf, False), 17)
            self.assertEqual(agents._alphabeta(TreeGame("l3"), 4, -math.inf, math.inf, True), 17)
        self.assertEqual(agents.alphabeta_agent({}, TreeGame(done=True)), Action.STAY)

    def test_invalid_depths_are_rejected(self):
        for depth in (0, -1, 1.5, True, "3"):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                agents.alphabeta_agent({}, TreeGame(), depth=depth)

    def test_timeout_before_first_depth_returns_legal_fallback(self):
        with patch.object(agents, "_check_search_deadline", side_effect=agents._SearchTimeout):
            self.assertEqual(agents.alphabeta_agent({}, TreeGame()), Action.LEFT)
        self.assertIsNone(agents._SEARCH_DEADLINE.get())

    def test_real_games_have_minimax_value_with_fewer_nodes(self):
        for seed in (0, 1, 7):
            random.seed(seed)
            game = PacmanGame(GameConfig(maze_seed=seed), render=False)
            for depth in (1, 2, 3):
                with self.subTest(seed=seed, depth=depth):
                    agents._search_stats["nodes"] = 0
                    expected = agents._minimax(game, depth, True)
                    minimax_nodes = agents._search_stats["nodes"]
                    agents._search_stats["nodes"] = 0
                    value = agents._alphabeta(game, depth, -math.inf, math.inf, True)
                    self.assertEqual(value, expected)
                    self.assertLessEqual(agents._search_stats["nodes"], minimax_nodes)
                    action = agents.alphabeta_agent({}, game, depth=depth)
                    self.assertEqual(root_value(game, action, depth), expected)

    def test_live_state_and_rng_remain_unchanged_and_result_repeats(self):
        game = scenario(ghosts=[(5, 3, GhostBehavior.DFS, True)], ammo=1)
        before, rng = game.get_state(), random.getstate()
        action = agents.alphabeta_agent({}, game, depth=3)
        self.assertIn(action, game.get_legal_actions())
        self.assertEqual(before, game.get_state())
        self.assertEqual(rng, random.getstate())
        self.assertEqual(action, agents.alphabeta_agent({}, game, depth=3))

    def test_real_game_kills_adjacent_final_ghost(self):
        game = scenario(ghosts=[(4, 3, GhostBehavior.FOLLOW, True)], ammo=1)
        action = agents.alphabeta_agent({}, game, depth=3)
        child = game.clone()
        child.step(action)
        self.assertTrue(child.player.alive)
        self.assertFalse(any(ghost.alive for ghost in child.ghosts))

    def test_ordering_puts_hits_first_for_max_and_last_for_min(self):
        game = scenario(ghosts=[(4, 4, GhostBehavior.FOLLOW, True)], ammo=1)
        actions = game.get_legal_actions()
        max_order = agents._order_actions(game, actions, maximising=True)
        min_order = agents._order_actions(game, actions, maximising=False)
        self.assertEqual(max_order[0], Action.SHOOT_DOWN_RIGHT)
        self.assertEqual(min_order[-1], Action.SHOOT_DOWN_RIGHT)
        self.assertEqual(sorted(max_order, key=lambda a: a.value),
                         sorted(actions, key=lambda a: a.value))


if __name__ == "__main__":
    unittest.main()
