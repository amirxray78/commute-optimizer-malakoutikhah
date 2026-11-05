"""Run: python -m unittest discover -s practical-2/tests -v (from repo root)."""

from copy import deepcopy
import math
import os
from pathlib import Path
import random
import sys
import unittest

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import GameConfig, GameState, GhostBehavior, PacmanGame
from student_assignment import evaluate_state


def scenario(ghosts=None, ammo=0, walls=(), traveled=0, alive=True, done=False,
             diagonal=True, steps=0, player=(3, 3)):
    config = GameConfig(grid_width=7, grid_height=7, num_ghosts=0,
                        maze_seed=42, diagonal_shooting=diagonal)
    game = PacmanGame(config=config, render=False)
    if ghosts is None:
        ghosts = [(6, 6, GhostBehavior.FOLLOW, True)]
    game.set_state(GameState(
        player_x=player[0], player_y=player[1], player_ammo=ammo,
        player_tiles_traveled=traveled, player_alive=alive,
        ghosts=[dict(x=x, y=y, behavior=behavior, alive=living)
                for x, y, behavior, living in ghosts],
        step_count=steps, score=0, done=done, walls=list(walls),
    ))
    return game


class EvaluationTests(unittest.TestCase):
    def test_death_is_worse_than_survival_even_after_kills(self):
        dead = scenario(ghosts=[(6, 6, GhostBehavior.FOLLOW, False)],
                        ammo=5, traveled=90, alive=False, done=True)
        self.assertLess(evaluate_state(dead), evaluate_state(scenario()))

    def test_clearing_ghosts_is_preferred_to_ongoing_play(self):
        clear = scenario(ghosts=[(6, 6, GhostBehavior.FOLLOW, False)])
        self.assertGreater(evaluate_state(clear), evaluate_state(scenario(ammo=10)))

    def test_kill_reward_outweighs_spending_one_ammo(self):
        before = scenario(ghosts=[(5, 5, GhostBehavior.RANDOM, True),
                                 (6, 6, GhostBehavior.RANDOM, True)], ammo=1)
        after = scenario(ghosts=[(5, 5, GhostBehavior.RANDOM, False),
                                (6, 6, GhostBehavior.RANDOM, True)], ammo=0)
        self.assertGreater(evaluate_state(after), evaluate_state(before))

    def test_unarmed_player_prefers_more_distance_to_ghost(self):
        close = scenario(ghosts=[(3, 2, GhostBehavior.FOLLOW, True)])
        far = scenario(ghosts=[(3, 0, GhostBehavior.FOLLOW, True)])
        self.assertGreater(evaluate_state(far), evaluate_state(close))

    def test_more_ammo_improves_same_position(self):
        values = [evaluate_state(scenario(ammo=ammo)) for ammo in (0, 1, 2)]
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_danger_accounts_for_ghost_behavior(self):
        values = [evaluate_state(scenario(ghosts=[(3, 0, behavior, True)]))
                  for behavior in (GhostBehavior.RANDOM, GhostBehavior.FOLLOW,
                                   GhostBehavior.LOOKAHEAD)]
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_distances_respect_walls_not_just_manhattan_distance(self):
        walls = {(2, 2), (2, 3), (2, 4)}
        behind_wall = scenario(ghosts=[(1, 3, GhostBehavior.FOLLOW, True)], walls=walls)
        open_path = scenario(ghosts=[(5, 3, GhostBehavior.FOLLOW, True)], walls=walls)
        self.assertGreater(evaluate_state(behind_wall), evaluate_state(open_path))

    def test_disconnected_ghost_does_not_cause_infinite_score(self):
        game = scenario(ghosts=[(1, 3, GhostBehavior.FOLLOW, True)],
                        walls={(2, y) for y in range(7)})
        self.assertTrue(math.isfinite(evaluate_state(game)))

    def test_mobility_prefers_open_position_to_trap(self):
        trap = scenario(walls={(3, 2), (3, 4), (2, 3), (4, 3)})
        self.assertGreater(evaluate_state(scenario()), evaluate_state(trap))

    def test_nearer_ammo_pickup_and_extra_steps_are_rewarded(self):
        self.assertGreater(evaluate_state(scenario(traveled=9, steps=9)),
                           evaluate_state(scenario(traveled=0, steps=0)))
        self.assertGreater(evaluate_state(scenario(ammo=1, traveled=10, steps=10)),
                           evaluate_state(scenario(ammo=0, traveled=9, steps=9)))

    def test_diagonal_shooting_opportunity_requires_ammo_and_enabled_rule(self):
        ghosts = [(4, 4, GhostBehavior.RANDOM, True)]
        self.assertGreater(evaluate_state(scenario(ghosts=ghosts, ammo=1, diagonal=True)),
                           evaluate_state(scenario(ghosts=ghosts, ammo=1, diagonal=False)))
        self.assertEqual(evaluate_state(scenario(ghosts=ghosts, ammo=0, diagonal=True)),
                         evaluate_state(scenario(ghosts=ghosts, ammo=0, diagonal=False)))

    def test_surviving_step_limit_has_terminal_bonus(self):
        self.assertGreater(evaluate_state(scenario(done=True, steps=100)),
                           evaluate_state(scenario(done=False, steps=100)))

    def test_evaluation_is_deterministic_and_read_only(self):
        game = scenario(ammo=1, traveled=3)
        before, config = game.get_state(), deepcopy(game.config)
        rng = random.getstate()
        first = evaluate_state(game)
        self.assertEqual(first, evaluate_state(game))
        self.assertEqual(before, game.get_state())
        self.assertEqual(config, game.config)
        self.assertEqual(rng, random.getstate())
        self.assertTrue(math.isfinite(first))


if __name__ == "__main__":
    unittest.main()
