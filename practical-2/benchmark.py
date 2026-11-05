"""Reproducible in-process benchmark for the three Pacman agents.

Every episode reseeds the global RNG, so all agents start from the same mazes
and ghost placements. Agents restore the RNG after searching, so an episode
changes only because of the actions an agent chooses. Each decision still uses
the agent's internal SEARCH_TIME_BUDGET, and the slowest decision is recorded
to check the 5-second limit.

    python benchmark.py                 # tuning grid + final comparison
    python benchmark.py --final-only    # final comparison with the selected settings
"""

from concurrent.futures import ProcessPoolExecutor
import csv
import os
from pathlib import Path
import random
import statistics
import sys
import time

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

from game import GameConfig, PacmanGame
import student_assignment as agents

RESULTS = Path(__file__).resolve().parent / 'results'
TUNING_SEEDS = range(100, 112)
FINAL_SEEDS = range(500, 530)

TUNING_GRID = [
    ('minimax', {'depth': 2}),
    ('minimax', {'depth': 3}),
    ('minimax', {'depth': 4}),
    ('alphabeta', {'depth': 4}),
    ('alphabeta', {'depth': 6}),
    ('alphabeta', {'depth': 8}),
    ('alphabeta', {'depth': 10}),
    ('mcts', {'iterations': 250, 'rollout_depth': 20, 'c': 1.41, 'policy': 'guided'}),
    ('mcts', {'iterations': 500, 'rollout_depth': 20, 'c': 1.41, 'policy': 'guided'}),
    ('mcts', {'iterations': 1000, 'rollout_depth': 20, 'c': 1.41, 'policy': 'guided'}),
    ('mcts', {'iterations': 500, 'rollout_depth': 10, 'c': 1.41, 'policy': 'guided'}),
    ('mcts', {'iterations': 500, 'rollout_depth': 40, 'c': 1.41, 'policy': 'guided'}),
    ('mcts', {'iterations': 500, 'rollout_depth': 20, 'c': 0.5, 'policy': 'guided'}),
    ('mcts', {'iterations': 500, 'rollout_depth': 20, 'c': 1.41, 'policy': 'random'}),
    # Round 2: around the best round-1 settings.
    ('alphabeta', {'depth': 5}),
    ('mcts', {'iterations': 500, 'rollout_depth': 5, 'c': 1.41, 'policy': 'guided'}),
    ('mcts', {'iterations': 1000, 'rollout_depth': 10, 'c': 1.41, 'policy': 'guided'}),
    ('mcts', {'iterations': 500, 'rollout_depth': 10, 'c': 2.5, 'policy': 'guided'}),
    ('mcts', {'iterations': 500, 'rollout_depth': 10, 'c': 1.41, 'policy': 'random'}),
]


def _decide(name, obs, game, params):
    if name == 'minimax':
        return agents.minimax_agent(obs, game, depth=params['depth'])
    if name == 'alphabeta':
        return agents.alphabeta_agent(obs, game, depth=params['depth'])
    agents.MCTS_ROLLOUT_POLICY = params['policy']
    return agents.mcts_agent(obs, game, iterations=params['iterations'],
                             rollout_depth=params['rollout_depth'], c=params['c'])


def run_episode(job):
    name, params, seed = job
    random.seed(seed)
    config = GameConfig(num_ghosts=5)  # default max_steps = 100
    game = PacmanGame(config=config, render=False)
    obs, info, times, nodes = game._obs(), game._info(), [], 0
    while not game.done:
        agents._search_stats['nodes'] = 0
        start = time.perf_counter()
        action = _decide(name, obs, game, params)
        times.append(time.perf_counter() - start)
        nodes += agents._search_stats['nodes']
        obs, _, _, info = game.step(action)
    kills = sum(not g.alive for g in game.ghosts)
    return {
        'agent': name, 'params': _label(params), 'seed': seed,
        'kills': kills, 'cleared': int(kills == config.num_ghosts),
        'survived': int(game.player.alive), 'steps': info['step'],
        'tiles': info['tiles_traveled'], 'score': info['score'],
        'mean_move_s': statistics.fmean(times), 'max_move_s': max(times),
        'search_nodes': nodes,
    }


def _label(params):
    return ' '.join(f'{k}={v}' for k, v in params.items())


def summarise(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['agent'], row['params']), []).append(row)
    summary = []
    for (agent, params), items in groups.items():
        summary.append({
            'agent': agent, 'params': params, 'episodes': len(items),
            **{f'avg_{k}': round(statistics.fmean(r[k] for r in items), 4)
               for k in ('kills', 'cleared', 'survived', 'steps', 'tiles',
                         'mean_move_s', 'search_nodes')},
            'max_move_s': round(max(r['max_move_s'] for r in items), 4),
        })
    return summary


def write_csv(path, rows):
    path.parent.mkdir(exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(grid, seeds, prefix):
    jobs = [(name, params, seed) for name, params in grid for seed in seeds]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = list(pool.map(run_episode, jobs))
    summary = summarise(rows)
    write_csv(RESULTS / f'{prefix}_runs.csv', rows)
    write_csv(RESULTS / f'{prefix}_summary.csv', summary)
    for row in summary:
        print(f"{row['agent']:<10} {row['params']:<58} kills={row['avg_kills']:.2f} "
              f"survived={row['avg_survived']:.2f} tiles={row['avg_tiles']:.1f} "
              f"steps={row['avg_steps']:.1f} max_move={row['max_move_s']:.2f}s", flush=True)
    return summary


FINAL_GRID = [
    ('minimax', {'depth': agents.MINIMAX_DEPTH}),
    ('alphabeta', {'depth': agents.ALPHABETA_DEPTH}),
    ('mcts', {'iterations': agents.MCTS_ITERATIONS,
              'rollout_depth': agents.MCTS_ROLLOUT_DEPTH,
              'c': agents.MCTS_C, 'policy': agents.MCTS_ROLLOUT_POLICY}),
]


if __name__ == '__main__':
    if '--final-only' not in sys.argv:
        print('--- tuning ---', flush=True)
        run(TUNING_GRID, TUNING_SEEDS, 'tuning')
    print('--- final comparison ---', flush=True)
    run(FINAL_GRID, FINAL_SEEDS, 'final')
