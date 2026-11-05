"""UCB1, MCTS tree bookkeeping, rollouts, and live-game isolation."""

import math
import os
from pathlib import Path
import random
import sys
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import Action, GameConfig, GhostBehavior, PacmanGame
import student_assignment as agents
from test_evaluation import scenario


class UCB1Tests(unittest.TestCase):
    def test_unvisited_node_is_infinite(self):
        self.assertEqual(agents.ucb1_score(0.0, 0, 10, 1.41), math.inf)

    def test_formula(self):
        self.assertAlmostEqual(agents.ucb1_score(0.5, 4, 100, 2.0),
                               0.5 + 2.0 * math.sqrt(math.log(100) / 4))

    def test_zero_exploration_is_mean_value(self):
        self.assertEqual(agents.ucb1_score(0.25, 3, 9, 0.0), 0.25)


class MCTSNodeTests(unittest.TestCase):
    def setUp(self):
        self.state = scenario().get_state()

    def test_expansion_status_and_list_copy(self):
        actions = [Action.UP, Action.STAY]
        node = agents.MCTSNode(self.state, untried_actions=actions)
        self.assertFalse(node.is_fully_expanded())
        node.untried_actions.pop()
        self.assertEqual(actions, [Action.UP, Action.STAY])
        node.untried_actions.clear()
        self.assertTrue(node.is_fully_expanded())

    def test_terminal_node_has_no_untried_actions(self):
        node = agents.MCTSNode(scenario(done=True).get_state(), untried_actions=[Action.UP])
        self.assertTrue(node.is_fully_expanded())

    def test_best_child_prefers_unvisited_then_ucb(self):
        root = agents.MCTSNode(self.state)
        root.visits = 10
        for visits, total in ((5, 4.0), (5, 1.0)):
            child = agents.MCTSNode(self.state, parent=root)
            child.visits, child.total_value = visits, total
            root.children.append(child)
        self.assertIs(root.best_child(1.0), root.children[0])
        fresh = agents.MCTSNode(self.state, parent=root)
        root.children.append(fresh)
        self.assertIs(root.best_child(1.0), fresh)

    def test_backprop_updates_every_ancestor(self):
        root = agents.MCTSNode(self.state)
        mid = agents.MCTSNode(self.state, parent=root)
        leaf = agents.MCTSNode(self.state, parent=mid)
        agents._mcts_backprop(leaf, 0.75)
        agents._mcts_backprop(mid, 0.25)
        self.assertEqual((root.visits, root.total_value), (2, 1.0))
        self.assertEqual((mid.visits, mid.total_value), (2, 1.0))
        self.assertEqual((leaf.visits, leaf.total_value), (1, 0.75))


class MCTSHelperTests(unittest.TestCase):
    def test_actions_drop_empty_shots_and_put_hits_first(self):
        game = scenario(ghosts=[(4, 4, GhostBehavior.FOLLOW, True)], ammo=1)
        actions = agents._mcts_actions(game)
        self.assertEqual(actions[0], Action.SHOOT_DOWN_RIGHT)
        self.assertEqual([a for a in actions if a.name.startswith("SHOOT")],
                         [Action.SHOOT_DOWN_RIGHT])
        self.assertIn(Action.STAY, actions)

    def test_select_descends_fully_expanded_nodes_and_restores_state(self):
        game = scenario(ammo=1)
        root = agents.MCTSNode(game.get_state(), untried_actions=[Action.UP])
        child = agents._mcts_expand(root, game)
        self.assertEqual(child.action_taken, Action.UP)
        self.assertEqual(root.children, [child])
        agents._mcts_backprop(child, 1.0)
        other = scenario(player=(0, 0))
        self.assertIs(agents._mcts_select(root, other), child)
        self.assertEqual(other.get_state(), child.state_snapshot)

    def test_expand_terminal_node_returns_itself(self):
        game = scenario(done=True)
        node = agents.MCTSNode(game.get_state())
        self.assertIs(agents._mcts_expand(node, game), node)

    def test_rollout_respects_depth_and_terminal_states(self):
        game = scenario(ghosts=[(6, 6, GhostBehavior.RANDOM, True)], steps=0)
        with patch.object(agents, "evaluate_state", return_value=5.0):
            self.assertEqual(agents._mcts_simulate(game, 4), 5.0)
        self.assertEqual(game.step_count, 4)
        done = scenario(done=True)
        agents._mcts_simulate(done, 10)
        self.assertEqual(done.step_count, 0)

    def test_random_rollout_policy_is_available(self):
        game = scenario(ghosts=[(6, 6, GhostBehavior.RANDOM, True)])
        with patch.object(agents, "MCTS_ROLLOUT_POLICY", "random"):
            agents._mcts_simulate(game, 3)
        self.assertEqual(game.step_count, 3)

    def test_guided_rollout_shoots_adjacent_ghost(self):
        game = scenario(ghosts=[(3, 2, GhostBehavior.FOLLOW, True)], ammo=1)
        with patch.object(agents, "MCTS_ROLLOUT_EPSILON", 0.0):
            self.assertEqual(agents._rollout_action(game), Action.SHOOT_UP)

    def test_guided_rollout_avoids_ghost_neighbourhood_without_ammo(self):
        game = scenario(ghosts=[(5, 3, GhostBehavior.FOLLOW, True)], ammo=0)
        with patch.object(agents, "MCTS_ROLLOUT_EPSILON", 0.0):
            for _ in range(30):
                self.assertNotEqual(agents._rollout_action(game), Action.RIGHT)


class MCTSAgentTests(unittest.TestCase):
    def test_terminal_game_returns_stay(self):
        self.assertEqual(agents.mcts_agent({}, scenario(done=True)), Action.STAY)

    def test_live_state_and_rng_remain_unchanged_and_result_repeats(self):
        game = scenario(ghosts=[(5, 3, GhostBehavior.DFS, True)], ammo=1)
        before, rng = game.get_state(), random.getstate()
        action = agents.mcts_agent({}, game, iterations=60, rollout_depth=8)
        self.assertIn(action, game.get_legal_actions())
        self.assertEqual(before, game.get_state())
        self.assertEqual(rng, random.getstate())
        self.assertEqual(action, agents.mcts_agent({}, game, iterations=60, rollout_depth=8))

    def test_kills_adjacent_final_ghost(self):
        game = scenario(ghosts=[(4, 3, GhostBehavior.FOLLOW, True)], ammo=1)
        self.assertEqual(agents.mcts_agent({}, game, iterations=80, rollout_depth=5),
                         Action.SHOOT_RIGHT)

    def test_avoids_death_without_ammo(self):
        game = scenario(ghosts=[(5, 3, GhostBehavior.FOLLOW, True)], ammo=0)
        action = agents.mcts_agent({}, game, iterations=150, rollout_depth=5)
        child = game.clone()
        child.step(action)
        self.assertTrue(child.player.alive)

    def test_time_budget_stops_large_iteration_counts(self):
        random.seed(3)
        game = PacmanGame(GameConfig(maze_seed=3), render=False)
        with patch.object(agents, "SEARCH_TIME_BUDGET", 0.3):
            start = time.perf_counter()
            action = agents.mcts_agent({}, game, iterations=10**9)
        self.assertLess(time.perf_counter() - start, 1.5)
        self.assertIn(action, game.get_legal_actions())

    def test_zero_iterations_returns_legal_fallback(self):
        game = scenario(ammo=0)
        self.assertIn(agents.mcts_agent({}, game, iterations=0), game.get_legal_actions())


if __name__ == "__main__":
    unittest.main()
