"""
student_assignment.py — AI assignment: Pacman-like game agents
==============================================================

Your task: implement three AI agents that play the Pacman-like game,
plus a shared evaluation function used by all of them.

Each agent receives the current game observation and a live PacmanGame
instance, and must return an Action.

Run this file to test:
    python student_assignment.py                 # human window
    python student_assignment.py minimax         # evaluate minimax (headless)
    python student_assignment.py alphabeta
    python student_assignment.py mcts
    python student_assignment.py alphabeta 10    # 10 episodes instead of 5
    python student_assignment.py alphabeta --depth 8   # override the depth
    python student_assignment.py run minimax     # watch agent play in window

Search depth / budget for every agent lives in the AGENT SETTINGS block
below — tune them there (or use --depth on the command line).

Actions (13 total — see the Action enum in game.py):
    Move:  UP, DOWN, LEFT, RIGHT
    Shoot: SHOOT_UP, SHOOT_DOWN, SHOOT_LEFT, SHOOT_RIGHT,
           SHOOT_UP_LEFT, SHOOT_UP_RIGHT, SHOOT_DOWN_LEFT, SHOOT_DOWN_RIGHT
    Stay:  STAY — player skips turn; ghosts still move.
           Returned automatically when your agent exceeds the 5-second time
           limit per move.
    A shot hits a ghost on the adjacent tile in that direction (diagonals
    included) and costs 1 ammo whether it hits or not. Diagonal shots can be
    turned off with GameConfig(diagonal_shooting=False).
    Always build your action list from game.get_legal_actions() rather than
    hard-coding it — shoot actions only appear when the player has ammo.

Useful game API:
    game.clone()             → independent copy for simulation (no window)
    game.get_state()         → GameState snapshot
    game.set_state(state)    → restore from snapshot
    game.get_legal_actions() → list[Action] valid this turn
    game.step(action)        → (obs, reward, done, info)
    game.done                → bool
    game.player.x/y/ammo/alive
    game.ghosts              → list[Ghost]  (each has .x .y .alive .behavior)

──────────────────────────────────────────────────────────────────────────────
Game State Reference  (everything evaluate_state / your agent can read)
──────────────────────────────────────────────────────────────────────────────

Player:
    p = game.player
    p.x, p.y            # current position (column, row)
    p.ammo              # shots remaining (starts at 1)
    p.alive             # bool
    p.tiles_traveled    # total tiles walked so far

Ghosts:
    game.ghosts                              # list[Ghost] — alive or dead
    alive_ghosts = [g for g in game.ghosts if g.alive]
    g.x, g.y, g.alive, g.behavior           # per ghost fields

    Ghost threat levels (GhostBehavior enum):
        GhostBehavior.RANDOM     # wanders — least dangerous
        GhostBehavior.FOLLOW     # BFS chases you directly
        GhostBehavior.LOOKAHEAD  # plans ahead — most dangerous

Walls:
    game.walls                   # set of (x, y) tuples — impassable tiles
    if (x, y) in game.walls: …  # check before moving

Grid size:
    game.config.grid_width   # columns
    game.config.grid_height  # rows

Ammo pickup timing (+1 ammo every tiles_per_ammo tiles walked, default 10):
    tiles_until_ammo = (
        game.config.tiles_per_ammo
        - p.tiles_traveled % game.config.tiles_per_ammo
    )
    # value == 1  → next step earns ammo
    # value == 10 → just earned ammo

Manhattan distance to a ghost (fast, no wall awareness):
    dist = abs(g.x - p.x) + abs(g.y - p.y)
    nearest = min(alive_ghosts, key=lambda g: abs(g.x-p.x)+abs(g.y-p.y))

Ghosts in shooting range (adjacent 8 tiles):
    in_range = [
        g for g in alive_ghosts
        if abs(g.x - p.x) <= 1 and abs(g.y - p.y) <= 1
        and (g.x != p.x or g.y != p.y)
    ]
    # shoot actions only appear in get_legal_actions() when p.ammo > 0

Reactive defence:
    If a ghost steps onto the player's tile and p.ammo > 0, the ghost dies
    automatically inside game.step() — no action needed.
──────────────────────────────────────────────────────────────────────────────
"""

import math
import multiprocessing as mp
import random
import sys
from typing import Dict, List, Optional

from game import Action, GameConfig, GameState, GhostBehavior, PacmanGame


# ──────────────────────────────────────────────────────────────────────────────
# Time-limited agent wrapper
# ──────────────────────────────────────────────────────────────────────────────

MOVE_TIME_LIMIT = 5.0

_timeout_counter: Dict[str, int] = {}


def _worker(agent_name: str, obs: Dict, state: GameState,
            config: GameConfig, kwargs: dict, q: mp.SimpleQueue) -> None:
    game = PacmanGame(config=config, render=False)
    game.set_state(state)
    fn = _AGENT_FNS[agent_name]
    try:
        q.put(fn(obs, game, **kwargs))
    except Exception:
        q.put(Action.STAY)


def _timed(agent_name: str, obs: Dict, game: PacmanGame, **kwargs) -> Action:
    q = mp.SimpleQueue()
    p = mp.Process(
        target=_worker,
        args=(agent_name, obs, game.get_state(), game.config, kwargs, q),
        daemon=True,
    )
    p.start()
    p.join(MOVE_TIME_LIMIT)
    if p.is_alive():
        p.kill()
        p.join()
        _timeout_counter[agent_name] = _timeout_counter.get(agent_name, 0) + 1
        return Action.STAY
    try:
        return q.get()
    except Exception:
        return Action.STAY


# ──────────────────────────────────────────────────────────────────────────────
# AGENT SETTINGS — tune these
# ──────────────────────────────────────────────────────────────────────────────
# "depth" means plies: one ply = one game.step() (player acts, ghosts react).
# The player has up to 12 legal actions, so a full tree has ~12^depth nodes.

MINIMAX_DEPTH      = 3     # plain minimax: every extra ply ≈ 12× slower
ALPHABETA_DEPTH    = 6     # pruning makes much deeper search affordable

MCTS_ITERATIONS    = 500   # selection → backprop cycles per move
MCTS_ROLLOUT_DEPTH = 20    # steps per rollout (simulation phase)
MCTS_C             = 1.41  # UCB1 exploration constant (C in Q + C*sqrt(ln(N)/n))


# ──────────────────────────────────────────────────────────────────────────────
# TODO 1 — Scoring / evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_state(game: PacmanGame) -> float:
    """
    Return a numeric score for the current game state (higher = better for player).

    This function is the heart of every agent below. Spend time here first.

    Suggested features to consider (you are NOT limited to these):
      • Distance to the nearest alive ghost (farther → safer → higher score?)
      • Number of alive ghosts remaining (fewer → better?)
      • Player ammo count and tiles_until_ammo (how close to the next free ammo)
      • Ghosts within shooting range — any of the 8 tiles around the player
        (including diagonals) can be hit if you have ammo
      • Survival: big bonus if player is alive, big penalty if dead
      • Step count / tiles traveled (reward staying alive longer?)
      • Mobility: BFS from player position — being trapped is dangerous even
        before any ghost arrives

    You can also give different weights to different ghost behaviors — a LOOKAHEAD
    ghost is more dangerous than a RANDOM ghost (see GhostBehavior enum).

    Args:
        game: PacmanGame instance (do NOT call game.step() here — read only).

    Returns:
        float score.

    TODO: replace the NotImplementedError with your implementation.
    """
    # ── placeholder ───────────────────────────────────────────────────────────
    raise NotImplementedError(
        "Implement evaluate_state(). "
        "This function is used by all four agents."
    )

    # ── example skeleton (remove the raise above to activate) ─────────────────
    # p = game.player
    # if not p.alive:
    #     return -1000.0
    #
    # alive_ghosts = [g for g in game.ghosts if g.alive]
    # if not alive_ghosts:
    #     return 1000.0
    #
    # # ── convenient pre-extracted values ───────────────────────────────────────
    # ghost_positions   = [(g.x, g.y) for g in alive_ghosts]   # list of (x, y)
    # walls             = game.walls                             # set of (x, y)
    # ammo              = p.ammo                                 # shots remaining
    # tiles_until_ammo  = (game.config.tiles_per_ammo
    #                      - p.tiles_traveled % game.config.tiles_per_ammo)
    # # tiles_until_ammo == game.config.tiles_per_ammo means ammo just refreshed
    # # ──────────────────────────────────────────────────────────────────────────
    #
    # nearest_dist = min(abs(gx - p.x) + abs(gy - p.y) for gx, gy in ghost_positions)
    #
    # score = 0.0
    # score += ???   # reward distance to nearest ghost
    # score += ???   # reward fewer alive ghosts
    # score += ???   # reward ammo / proximity to next ammo pickup
    # score += ???   # survival bonus
    # return score


# ──────────────────────────────────────────────────────────────────────────────
# Agent 1 — Minimax
# ──────────────────────────────────────────────────────────────────────────────

def minimax_agent(obs: Dict, game: PacmanGame,
                  depth: int = MINIMAX_DEPTH) -> Action:
    """
    Minimax search agent.

    The player (MAX node) tries to maximise evaluate_state().
    The ghosts collectively act as the MIN player and try to minimise it.

    Args:
        obs:   Current observation dict (from game.reset / game.step).
        game:  Live PacmanGame — use game.clone() before simulating.
        depth: Ply depth (number of alternating MAX/MIN levels).

    Returns:
        The Action with the highest minimax value.

    TODO:
      1. For each legal action:
           a. Clone the game.
           b. Apply the action with clone.step(action).
           c. Call _minimax(clone, depth-1, maximising=False).
           d. Track the best-valued action.
      2. Return the best action.

    Note: with ammo the player has up to 12 legal actions (4 moves + 8 shots),
    so the tree grows ~12^depth. Keep MINIMAX_DEPTH small — Agent 2
    (alpha-beta) is where you go deep.
    """
    raise NotImplementedError("Implement minimax_agent().")


def _minimax(game: PacmanGame, depth: int, maximising: bool) -> float:
    """
    Recursive minimax.

    TODO:
      • Base case: return evaluate_state(game) when depth == 0 or game.done.
      • MAX turn: return max score over legal actions (player's turn).
      • MIN turn: return min score over legal actions (ghost-optimal counter).
        (Simplification: treat ghosts as a single adversary choosing the worst
         action for the player.)
    """
    raise NotImplementedError("Implement _minimax().")


# ──────────────────────────────────────────────────────────────────────────────
# Agent 2 — Minimax with alpha-beta pruning
# ──────────────────────────────────────────────────────────────────────────────

def alphabeta_agent(obs: Dict, game: PacmanGame,
                    depth: int = ALPHABETA_DEPTH) -> Action:
    """
    Minimax with alpha-beta pruning.

    Returns exactly the same action as minimax_agent at the same depth, but
    skips branches that cannot change the result — so it can search much
    deeper (ALPHABETA_DEPTH vs MINIMAX_DEPTH) in the same time.

    The two bounds, carried down the tree:
        alpha — the best score MAX (player) is already guaranteed elsewhere
        beta  — the best score MIN (ghosts) is already guaranteed elsewhere
    As soon as alpha >= beta at a node, the opponent would never allow play to
    reach it, so the rest of its children can be skipped (pruned).

    Args:
        obs:   Current observation dict.
        game:  Live PacmanGame — use game.clone() before simulating.
        depth: Ply depth to search.

    Returns:
        The Action with the highest minimax value.

    TODO:
      1. alpha = -inf, beta = +inf, best_action = None.
      2. For each legal action (order them — see hint below):
           a. Clone the game and clone.step(action).
           b. value = _alphabeta(clone, depth - 1, alpha, beta, maximising=False)
           c. If value beats the best so far, remember this action.
           d. alpha = max(alpha, value)   ← lets later siblings prune earlier
      3. Return best_action.

    Hint — move ordering matters a lot. Pruning is strongest when the best
    move is searched first. Cheap ideas: try shots at adjacent ghosts first,
    or sort children by evaluate_state() after one step (best-first for MAX,
    worst-first for MIN).

    Hint — if a move takes too long, lower ALPHABETA_DEPTH, or try iterative
    deepening: search depth 1, 2, 3, … until a time limit and keep the result
    of the last completed depth (it also gives you a great move ordering).
    """
    raise NotImplementedError("Implement alphabeta_agent().")


def _alphabeta(game: PacmanGame, depth: int, alpha: float, beta: float,
               maximising: bool) -> float:
    """
    Recursive minimax with alpha-beta pruning.

    TODO:
      • Base case: return evaluate_state(game) when depth == 0 or game.done.
      • MAX node (player):
            value = -inf
            for each legal action:
                recurse on a stepped clone with (depth - 1, alpha, beta, False)
                value = max(value, child value)
                alpha = max(alpha, value)
                if alpha >= beta: stop looping   ← beta cut-off
            return value
      • MIN node (ghosts): the mirror image — minimise, tighten beta,
        and stop looping when beta <= alpha (alpha cut-off).

    Sanity check: at the same depth, alphabeta_agent and minimax_agent must
    pick moves with the same value. Count visited nodes in both to see how
    much pruning saves.
    """
    raise NotImplementedError("Implement _alphabeta().")


# ──────────────────────────────────────────────────────────────────────────────
# TODO 2 — UCB1 score formula
# ──────────────────────────────────────────────────────────────────────────────

def ucb1_score(q: float, n: int, N: int, c: float) -> float:
    """
    Slide 42 — UCB1 formula:

        UCB1(node) = U(n)/N(n)  +  C × sqrt( log N(PARENT(n)) / N(n) )

    Translated to this function's parameters:
        q = U(n)/N(n)  — mean rollout value of this node  (total_value / visits)
        n = N(n)       — visit count of this node
        N = N(PARENT)  — visit count of the parent node
        c = C          — exploration constant

    Returns:
        float UCB1 score (higher = prefer this node).

    Notes:
      • n == 0 means the node is unvisited — return +infinity so it is always
        chosen before any visited sibling (forces every child to be tried once).
      • Called by MCTSNode.best_child() for every child in the selection step.

    TODO: replace the NotImplementedError with your implementation.
    """
    raise NotImplementedError("Implement ucb1_score().")


# ──────────────────────────────────────────────────────────────────────────────
# Agent 3 — Monte Carlo Tree Search with UCB1 (UCT)
# ──────────────────────────────────────────────────────────────────────────────

class MCTSNode:
    """
    One node in the MCTS search tree.

    Each node represents a game state reached by one specific action from
    its parent.  The tree is grown incrementally: a node starts with all
    actions untried; expansion adds one child per call.

    Attributes you will need:
        state_snapshot  — GameState saved at this node (use game.get_state())
        parent          — parent MCTSNode (None for root)
        action_taken    — Action that led from parent to this node
        children        — list[MCTSNode] expanded so far
        visits          — how many times this node has been visited
        total_value     — cumulative rollout value (for mean calculation)
        untried_actions — actions not yet expanded into children

    TODO: complete is_fully_expanded() and best_child().
    """

    def __init__(self, state_snapshot: GameState,
                 parent: Optional['MCTSNode'] = None,
                 action_taken: Optional[Action] = None,
                 untried_actions: Optional[List[Action]] = None):
        self.state_snapshot  = state_snapshot
        self.parent          = parent
        self.action_taken    = action_taken
        self.children:       List['MCTSNode'] = []
        self.visits:         int   = 0
        self.total_value:    float = 0.0
        # TODO: populate untried_actions from legal moves at this state
        self.untried_actions: List[Action] = untried_actions or []

    def is_fully_expanded(self) -> bool:
        """Return True when every legal action has a child node."""
        # TODO: implement
        raise NotImplementedError("Implement MCTSNode.is_fully_expanded().")

    def best_child(self, c: float = 1.41) -> 'MCTSNode':
        """
        Return child with highest UCB1 score.

        TODO: call ucb1_score(child.mean_value(), child.visits, self.visits, c)
        for each child and return the one with the maximum score.
        """
        raise NotImplementedError("Implement MCTSNode.best_child().")

    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0


def mcts_agent(obs: Dict, game: PacmanGame,
               iterations:    int   = MCTS_ITERATIONS,
               rollout_depth: int   = MCTS_ROLLOUT_DEPTH,
               c:             float = MCTS_C) -> Action:
    """
    MCTS Version 2.0: UCT  (slide 43)

    Repeat `iterations` times:
      1. Recursively apply UCB to choose a path down to a leaf
         (not fully expanded) node n  →  _mcts_select()
      2. Add a new child c to n and run a rollout from c
         →  _mcts_expand() then _mcts_simulate()
      3. Update the win counts from c back up to the root
         →  _mcts_backprop()

    Then: choose the action leading to the child with highest N (visit count).

    Args:
        obs:           Current observation.
        game:          Live PacmanGame.
        iterations:    How many times to repeat the loop above.
        rollout_depth: Max steps in the rollout (simulation) phase.
        c:             UCB1 exploration constant (slide 42, the C in the formula).

    Returns:
        Action of the root child with highest visit count N.

    TODO:
      1. Build root MCTSNode from game.get_state() and game.get_legal_actions().
      2. Clone the game once for simulation (sim = game.clone()).
      3. For each iteration run steps 1-3 above using the helper functions.
      4. Return the action of root.children with max visits.
    """
    raise NotImplementedError("Implement mcts_agent().")


# ── MCTS helper functions (implement each one) ────────────────────────────────

def _mcts_select(root: MCTSNode, game: PacmanGame) -> MCTSNode:
    """
    Slide 43, step 1: "recursively apply UCB to choose a path down to a
    leaf (not fully expanded) node n".

    Start at root. At each node: if it is fully expanded AND not terminal,
    use best_child() (UCB1) to descend one level. Stop when you reach a node
    that still has untried actions (not fully expanded) or the game is over.

    Remember to restore game state at each node via game.set_state().

    TODO: implement the selection loop.
    """
    raise NotImplementedError("Implement _mcts_select().")


def _mcts_expand(node: MCTSNode, game: PacmanGame) -> MCTSNode:
    """
    Slide 43, step 2a: "add a new child c to n".

    Pop one action from node.untried_actions, apply it to the game,
    create a child MCTSNode for the resulting state, append it to
    node.children, and return it.

    TODO: implement expansion.
    """
    raise NotImplementedError("Implement _mcts_expand().")


def _mcts_simulate(game: PacmanGame, depth: int) -> float:
    """
    Slide 43, step 2b: "run a rollout from c".

    Play random actions for up to `depth` steps (stop early if game.done).
    Return evaluate_state(game) as the rollout value.

    TODO: implement the rollout.
    """
    raise NotImplementedError("Implement _mcts_simulate().")


def _mcts_backprop(node: MCTSNode, value: float):
    """
    Slide 43, step 3: "update the win counts from c back up to the root".

    Walk from node up to root (via node.parent), incrementing visits by 1
    and adding value to total_value at every ancestor including node itself.

    TODO: implement backpropagation.
    """
    raise NotImplementedError("Implement _mcts_backprop().")


# ──────────────────────────────────────────────────────────────────────────────
# Test harness
# ──────────────────────────────────────────────────────────────────────────────

_AGENT_FNS = {
    'minimax':   minimax_agent,
    'alphabeta': alphabeta_agent,
    'mcts':      mcts_agent,
}

_AGENTS = {
    'minimax':   lambda obs, g: _timed('minimax',   obs, g, depth=MINIMAX_DEPTH),
    'alphabeta': lambda obs, g: _timed('alphabeta', obs, g, depth=ALPHABETA_DEPTH),
    'mcts':      lambda obs, g: _timed('mcts',      obs, g,
                                       iterations=MCTS_ITERATIONS,
                                       rollout_depth=MCTS_ROLLOUT_DEPTH,
                                       c=MCTS_C),
}

# which setting `--depth N` overrides for each agent
_DEPTH_SETTING = {
    'minimax':   'MINIMAX_DEPTH',
    'alphabeta': 'ALPHABETA_DEPTH',
    'mcts':      'MCTS_ROLLOUT_DEPTH',
}


def evaluate_agent(agent_name: str,
                   num_episodes: int             = 5,
                   config: Optional[GameConfig]  = None) -> Dict:
    config = config or GameConfig(num_ghosts=5, max_steps=500)
    game   = PacmanGame(config=config, render=False)
    fn     = _AGENTS[agent_name]

    _timeout_counter[agent_name] = 0
    scores, lengths, kills, tiles = [], [], [], []
    for ep in range(num_episodes):
        before = _timeout_counter.get(agent_name, 0)
        obs    = game.reset()
        done   = False
        while not done:
            action = fn(obs, game)
            obs, _, done, info = game.step(action)
        ep_timeouts = _timeout_counter.get(agent_name, 0) - before
        scores.append(info['score'])
        lengths.append(info['step'])
        kills.append(config.num_ghosts - info['ghosts_alive'])
        tiles.append(info['tiles_traveled'])
        print(f"  episode {ep+1}: score={info['score']:.1f}  "
              f"steps={info['step']}  kills={config.num_ghosts - info['ghosts_alive']}  "
              f"tiles={info['tiles_traveled']}  timeouts={ep_timeouts}")

    total_timeouts = _timeout_counter.get(agent_name, 0)
    return {
        'agent':          agent_name,
        'avg_score':      sum(scores)  / len(scores),
        'avg_length':     sum(lengths) / len(lengths),
        'avg_kills':      sum(kills)   / len(kills),
        'avg_tiles':      sum(tiles)   / len(tiles),
        'total_timeouts': total_timeouts,
    }


def run_agent_in_window(agent_name: str,
                        config: Optional[GameConfig] = None) -> Dict:
    config = config or GameConfig(num_ghosts=5)
    game   = PacmanGame(config=config, render=True)
    fn     = _AGENTS[agent_name]
    _timeout_counter[agent_name] = 0

    def _agent_fn(obs):
        before = _timeout_counter.get(agent_name, 0)
        action = fn(obs, game)
        if _timeout_counter.get(agent_name, 0) > before:
            game.agent_timeouts += 1
        return action

    game.run(agent_fn=_agent_fn)
    kills = sum(1 for g in game.ghosts if not g.alive)
    return {'kills': kills, 'tiles_traveled': game.player.tiles_traveled,
            'timeouts': game.agent_timeouts}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _ask(prompt: str, options: list) -> int:
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"{prompt}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  Enter a number between 1 and {len(options)}.")


if __name__ == '__main__':
    headless      = '--headless'      in sys.argv
    no_time_limit = '--no-time-limit' in sys.argv
    assess        = '--assess'        in sys.argv

    if no_time_limit:
        _AGENTS['minimax']   = lambda obs, g: minimax_agent(obs, g, depth=MINIMAX_DEPTH)
        _AGENTS['alphabeta'] = lambda obs, g: alphabeta_agent(obs, g, depth=ALPHABETA_DEPTH)
        _AGENTS['mcts']      = lambda obs, g: mcts_agent(obs, g,
                                                         iterations=MCTS_ITERATIONS,
                                                         rollout_depth=MCTS_ROLLOUT_DEPTH,
                                                         c=MCTS_C)

    if assess:
        try:
            raw = input("Episodes per agent [5]: ").strip()
        except EOFError:
            raw = ''
        episodes = int(raw) if raw.isdigit() and int(raw) > 0 else 5
        all_agents = ['minimax', 'alphabeta', 'mcts']
        results = []
        for a in all_agents:
            print(f"\n--- {a} ({episodes} episodes) ---")
            results.append(evaluate_agent(a, num_episodes=episodes))
        col = 14
        print(f"\n{'agent':<{col}} {'avg_kills':>10} {'avg_tiles':>10} {'avg_score':>10} {'avg_steps':>10} {'timeouts':>10}")
        print('-' * (col + 54))
        for r in results:
            print(f"{r['agent']:<{col}} {r['avg_kills']:>10.2f} {r['avg_tiles']:>10.1f}"
                  f" {r['avg_score']:>10.1f} {r['avg_length']:>10.1f} {r['total_timeouts']:>10}")
        sys.exit(0)

    print("=== Pacman AI ===\n")

    agent_options = [
        'Human (you play)',
        f'Minimax            (depth={MINIMAX_DEPTH})',
        f'Alpha-Beta         (depth={ALPHABETA_DEPTH})',
        f'MCTS               (iter={MCTS_ITERATIONS}, rollout={MCTS_ROLLOUT_DEPTH})',
    ]
    agent_keys = ['human', 'minimax', 'alphabeta', 'mcts']
    agent = agent_keys[_ask("Select agent", agent_options)]

    if agent == 'human':
        print("\nControls: arrows/WASD move  Shift+dir shoot  R reset  Q quit")
        game = PacmanGame(render=True)
        game.run()
        kills = sum(1 for g in game.ghosts if not g.alive)
        print(f"\nkills: {kills}  tiles traveled: {game.player.tiles_traveled}")

    elif headless:
        raw = input("\nEpisodes [5]: ").strip()
        episodes = int(raw) if raw.isdigit() and int(raw) > 0 else 5
        print()
        stats = evaluate_agent(agent, num_episodes=episodes)
        print()
        for k, v in stats.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    else:
        stats = run_agent_in_window(agent)
        print(f"\nkills: {stats['kills']}  tiles traveled: {stats['tiles_traveled']}  timeouts: {stats['timeouts']}")
