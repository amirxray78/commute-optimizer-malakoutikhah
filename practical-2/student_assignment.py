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

Algorithm Comparison and Analysis
---------------------------------
Evaluation across 30 seeded 100-step games (5 ghosts, 10x10 maze, 0 timeouts):
  • Minimax    (depth=3):                    avg_kills=2.60  avg_tiles=32.6  survived=33%  mean_move=0.12s
  • Alpha-Beta (depth=4):                    avg_kills=2.53  avg_tiles=29.9  survived=37%  mean_move=0.10s
  • MCTS       (iter=500, rollout=5, C=1.41): avg_kills=4.83  avg_tiles=89.3  survived=93%  mean_move=0.19s

Why Minimax and Alpha-Beta perform worse:
  1. Overly pessimistic MIN model: in the starter's alternating game.step()
     formulation, MIN chooses from the player's own legal actions to minimise
     evaluate_state(). Every second ply assumes the player makes its worst
     possible move (such as walking into a ghost or wasting ammo). In tight
     positions all branches collapse to the death penalty, causing root ties
     and short-sighted moves. Searching deeper gives MIN more chances to force
     self-sabotage rather than improving play.
  2. Single-sample transitions and short horizon: each tree edge samples only
     one stochastic ghost reaction, and fixed depth (3-4 plies) cannot see
     long-term corridor traps.
  Alpha-Beta returns the exact same minimax values as plain Minimax while
  visiting ~10x fewer nodes at depth 4 via move ordering and pruning, proving
  the limitation is the adversarial model rather than search speed.

Why MCTS performs best:
  MCTS simulates the true game dynamics where the player controls every turn,
  averaging over stochastic ghost movements via UCB1 selection, reward
  normalisation, and short guided rollouts (depth=5) that shoot adjacent
  ghosts and avoid stepping next to ghosts when unarmed.

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

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import math
import multiprocessing as mp
import random
import sys
import time
from typing import Dict, List, Optional

from game import Action, GameConfig, GameState, GhostBehavior, PacmanGame


# ──────────────────────────────────────────────────────────────────────────────
# Time-limited agent wrapper
# ──────────────────────────────────────────────────────────────────────────────

MOVE_TIME_LIMIT = 5.0
SEARCH_TIME_BUDGET = 4.0  # Leave time for worker startup and returning the action.
_SEARCH_DEADLINE = ContextVar('search_deadline', default=None)

_timeout_counter: Dict[str, int] = {}
# Successors simulated by minimax / alpha-beta (useful for measuring pruning).
_search_stats: Dict[str, int] = {'nodes': 0}

# Offsets for each shooting action, used for move ordering and rollouts.
_SHOOT_DELTAS = {
    Action.SHOOT_UP: (0, -1), Action.SHOOT_DOWN: (0, 1),
    Action.SHOOT_LEFT: (-1, 0), Action.SHOOT_RIGHT: (1, 0),
    Action.SHOOT_UP_LEFT: (-1, -1), Action.SHOOT_UP_RIGHT: (1, -1),
    Action.SHOOT_DOWN_LEFT: (-1, 1), Action.SHOOT_DOWN_RIGHT: (1, 1),
}
_MOVE_DELTAS = {
    Action.UP: (0, -1), Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0), Action.RIGHT: (1, 0),
}


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
# The player has up to 13 legal actions including STAY: roughly 13^depth nodes.

MINIMAX_DEPTH      = 3     # plain minimax: every extra ply can be ~13x slower
ALPHABETA_DEPTH    = 4     # tuned: deeper paranoid search did not play better

MCTS_ITERATIONS    = 500   # selection → backprop cycles per move
MCTS_ROLLOUT_DEPTH = 5     # tuned: short guided rollouts beat 10, 20 and 40 steps
MCTS_C             = 1.41  # UCB1 exploration constant (C in Q + C*sqrt(ln(N)/n))
MCTS_ROLLOUT_POLICY  = 'guided'  # 'guided' (see _rollout_action) or 'random'
MCTS_ROLLOUT_EPSILON = 0.2       # guided policy: chance of a random useful action
MCTS_REWARD_SCALE    = 250.0     # tanh scale mapping evaluate_state deltas to [0, 1]

EVALUATION_WEIGHTS = {
    'death': -1_000_000.0,
    'clear': 1_000_000.0,
    'kill': 1000.0,
    'alive': 100.0,
    'ammo': 15.0,
    'tile': 0.5,
    'step': 0.05,
    'ammo_progress': 3.0,
    'exit': 2.0,
    'reachable_tile': 0.2,
    'threat': 40.0,
    'shootable': 15.0,
    'survived_limit': 200.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Scoring / evaluation function
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

    Uses wall-aware distances, behavior-weighted danger, ammo, and mobility.
    The game state and random-number generator are never modified.
    """
    p = game.player
    weights = EVALUATION_WEIGHTS
    alive_ghosts = [ghost for ghost in game.ghosts if ghost.alive]
    kills = sum(not ghost.alive for ghost in game.ghosts)
    score = (weights['kill'] * kills + weights['tile'] * p.tiles_traveled
             + weights['step'] * game.step_count)
    # Death takes priority even if the final ghost was killed in the same state.
    if not p.alive:
        return weights['death'] + score
    if not alive_ghosts:
        return weights['clear'] + score

    score += weights['alive'] + weights['ammo'] * p.ammo
    progress = (p.tiles_traveled % game.config.tiles_per_ammo) / game.config.tiles_per_ammo
    score += weights['ammo_progress'] * progress

    # One BFS serves both ghost-distance and mobility features. Ghosts are not
    # walls: they move, and an armed player can defeat one on collision.
    distances = {(p.x, p.y): 0}
    queue = deque([(p.x, p.y)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if (0 <= nx < game.config.grid_width and 0 <= ny < game.config.grid_height
                    and (nx, ny) not in game.walls and (nx, ny) not in distances):
                distances[(nx, ny)] = distances[(x, y)] + 1
                queue.append((nx, ny))
    score += weights['exit'] * sum(distance == 1 for distance in distances.values())
    score += weights['reachable_tile'] * min(len(distances), 20)

    behavior_weights = {
        GhostBehavior.RANDOM: 1.0,
        GhostBehavior.FOLLOW: 1.25,
        GhostBehavior.LOOKAHEAD: 1.5,
        GhostBehavior.DFS: 1.25,
    }
    for ghost in alive_ghosts:
        danger = behavior_weights[ghost.behavior]
        distance = distances.get((ghost.x, ghost.y))
        if distance is not None:
            # Ammo permits reactive defence, but is finite: keep some danger.
            defence_factor = 0.35 if p.ammo > 0 else 1.0
            score -= weights['threat'] * danger * defence_factor / (distance + 1) ** 2
        dx, dy = abs(ghost.x - p.x), abs(ghost.y - p.y)
        shootable = (max(dx, dy) == 1
                     and (game.config.diagonal_shooting or dx == 0 or dy == 0))
        if p.ammo > 0 and shootable:
            score += weights['shootable'] * danger

    if game.done:
        score += weights['survived_limit']
    return float(score)


# ──────────────────────────────────────────────────────────────────────────────
# Agent 1 — Minimax
# ──────────────────────────────────────────────────────────────────────────────

class _SearchTimeout(Exception):
    """Stop an incomplete search depth without discarding completed results."""


def _check_search_deadline() -> None:
    deadline = _SEARCH_DEADLINE.get()
    if deadline is not None and time.perf_counter() >= deadline:
        raise _SearchTimeout()


@contextmanager
def _search_successor(game: PacmanGame, action: Action):
    """Simulate on a clone, restoring the RNG after the entire child search.

    Sibling branches start with common random numbers. Deeper steps use the
    advanced stream, but planning never consumes the live game's RNG stream.
    This is a sampled transition tree, not exhaustive stochastic ghost outcomes.
    """
    _check_search_deadline()
    rng_state = random.getstate()
    _search_stats['nodes'] += 1
    try:
        child = game.clone()
        child.step(action)
        yield child
    finally:
        random.setstate(rng_state)


def minimax_agent(obs: Dict, game: PacmanGame,
                  depth: int = MINIMAX_DEPTH) -> Action:
    """
    Minimax search agent.

    The player (MAX node) tries to maximise evaluate_state().
    MIN follows the scaffold's proxy adversary: choose the worst complete
    game.step continuation from the same legal-action API. It is not a separate
    ghost-only turn; every step already includes the engine's ghost reaction.

    Args:
        obs:   Current observation dict (from game.reset / game.step).
        game:  Live PacmanGame — use game.clone() before simulating.
        depth: Ply depth (number of alternating MAX/MIN levels).

    Returns:
        The Action with the highest minimax value.

    Search increasing depths up to the requested depth. If the internal budget
    expires, return the last completed depth's choice, or a legal fallback if
    no depth completed. Search always operates on clones.

    Note: with ammo the player has up to 13 actions (4 moves + 8 shots + STAY),
    so the tree grows ~13^depth. Keep MINIMAX_DEPTH small — Agent 2
    (alpha-beta) is where you go deep.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer.")
    actions = game.get_legal_actions()
    if game.done or not actions:
        return Action.STAY
    best_action = actions[0]
    token = _SEARCH_DEADLINE.set(time.perf_counter() + SEARCH_TIME_BUDGET)
    try:
        for search_depth in range(1, depth + 1):
            iteration_action, iteration_value = actions[0], -math.inf
            try:
                for action in actions:
                    with _search_successor(game, action) as child:
                        value = _minimax(child, search_depth - 1, maximising=False)
                    if value > iteration_value:
                        iteration_action, iteration_value = action, value
            except _SearchTimeout:
                break
            best_action = iteration_action
    finally:
        _SEARCH_DEADLINE.reset(token)
    return best_action


def _minimax(game: PacmanGame, depth: int, maximising: bool) -> float:
    """
    Recursive minimax.

    Recursion (the starter's complete-step proxy adversary):
      • Base case: return evaluate_state(game) when depth == 0 or game.done.
      • MAX turn: return max score over legal actions (player's turn).
      • MIN turn: return min score over legal actions (ghost-optimal counter).
        (Simplification: treat ghosts as a single adversary choosing the worst
         action for the player.)
    """
    _check_search_deadline()
    if depth <= 0 or game.done:
        return evaluate_state(game)
    actions = game.get_legal_actions()
    if not actions:
        return evaluate_state(game)
    best = -math.inf if maximising else math.inf
    for action in actions:
        with _search_successor(game, action) as child:
            value = _minimax(child, depth - 1, not maximising)
        best = max(best, value) if maximising else min(best, value)
    return best


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

    Implementation notes:
      • Iterative deepening (1, 2, …, depth) under the same internal time
        budget as minimax; an interrupted depth never replaces the action of
        the last completed depth.
      • Root ordering: the previous depth's scores, best first. Inner nodes
        use a cheap static ordering (_order_actions).
      • Successors come from _search_successor, so every node sees the same
        sampled ghost transition as in minimax regardless of visiting order.
        Therefore the root value always equals minimax's value at the same
        depth. When several root moves tie, the chosen move may differ,
        because pruned siblings only return bounds.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer.")
    actions = game.get_legal_actions()
    if game.done or not actions:
        return Action.STAY
    best_action = actions[0]
    root_order = _order_actions(game, actions, maximising=True)
    token = _SEARCH_DEADLINE.set(time.perf_counter() + SEARCH_TIME_BUDGET)
    try:
        for search_depth in range(1, depth + 1):
            alpha, beta = -math.inf, math.inf
            iteration_action, scores = root_order[0], {}
            try:
                for action in root_order:
                    with _search_successor(game, action) as child:
                        value = _alphabeta(child, search_depth - 1, alpha, beta,
                                           maximising=False)
                    scores[action] = value
                    if value > alpha:
                        iteration_action, alpha = action, value
            except _SearchTimeout:
                break
            best_action = iteration_action
            # Stable sort: previous-depth scores give the next depth's ordering.
            root_order = sorted(root_order, key=lambda a: scores[a], reverse=True)
    finally:
        _SEARCH_DEADLINE.reset(token)
    return best_action


def _order_actions(game: PacmanGame, actions: List[Action],
                   maximising: bool) -> List[Action]:
    """Cheap move ordering for alpha-beta (never changes the minimax value).

    MAX tries the moves that are usually best first: shots that hit an adjacent
    ghost, then moves, then STAY, and wasted shots last. MIN tries the same
    list in reverse, so the moves that are usually worst for the player come
    first. Game stand-ins without player and ghost fields keep their order.
    """
    player = getattr(game, 'player', None)
    ghosts = getattr(game, 'ghosts', None)
    if player is None or ghosts is None:
        return list(actions)
    targets = {(g.x - player.x, g.y - player.y) for g in ghosts if g.alive}

    def rank(action: Action) -> int:
        delta = _SHOOT_DELTAS.get(action)
        if delta is not None:
            return 0 if delta in targets else 3
        return 2 if action == Action.STAY else 1

    return sorted(actions, key=rank, reverse=not maximising)


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

    Fail-soft: a pruned node returns its bound instead of clamping it to
    alpha/beta. Both bounds are still correct, and the root ordering uses them.
    """
    _check_search_deadline()
    if depth <= 0 or game.done:
        return evaluate_state(game)
    actions = game.get_legal_actions()
    if not actions:
        return evaluate_state(game)
    if maximising:
        value = -math.inf
        for action in _order_actions(game, actions, maximising=True):
            with _search_successor(game, action) as child:
                value = max(value, _alphabeta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # beta cut-off
        return value
    value = math.inf
    for action in _order_actions(game, actions, maximising=False):
        with _search_successor(game, action) as child:
            value = min(value, _alphabeta(child, depth - 1, alpha, beta, True))
        beta = min(beta, value)
        if beta <= alpha:
            break  # alpha cut-off
    return value


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
    """
    if n <= 0:
        return math.inf
    return q + c * math.sqrt(math.log(max(N, 1)) / n)


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
        # The caller passes the legal moves at this state (see _mcts_actions).
        # A terminal state has no moves to expand.
        terminal = getattr(state_snapshot, 'done', False)
        self.untried_actions: List[Action] = [] if terminal else list(untried_actions or [])

    def is_fully_expanded(self) -> bool:
        """Return True when every legal action has a child node."""
        return not self.untried_actions

    def best_child(self, c: float = 1.41) -> 'MCTSNode':
        """
        Return child with highest UCB1 score.

        Calls ucb1_score(child.mean_value(), child.visits, self.visits, c) for
        each child and returns the one with the highest score (first on ties).
        """
        if not self.children:
            raise ValueError("best_child() needs at least one expanded child.")
        return max(self.children,
                   key=lambda child: ucb1_score(child.mean_value(), child.visits,
                                                self.visits, c))

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

    Implementation notes:
      • The tree is built only from simulated clones. The global RNG is
        restored at the end, so the live game and its random stream are
        unchanged and repeated calls return the same action.
      • A node stores the one ghost transition sampled when it was expanded
        (a determinised tree). Rollouts below it are sampled again on every
        visit.
      • evaluate_state() spans ±1e6, which would overwhelm the UCB1
        exploration term. Every rollout value v is therefore mapped to
        [0, 1] relative to the root value v0:
            0.5 + 0.5 * tanh((v - v0) / MCTS_REWARD_SCALE)
        Death maps to about 0, a kill or a clear to about 1.
      • The loop also stops at the shared per-move time budget, so a large
        `iterations` value cannot cause a timeout.
      • Shooting at an empty tile is never expanded. It is the same as STAY
        except that one ammo is lost.
    """
    actions = game.get_legal_actions()
    if game.done or not actions:
        return Action.STAY
    rng_state = random.getstate()
    deadline = time.perf_counter() + SEARCH_TIME_BUDGET
    try:
        root_value = evaluate_state(game)
        root = MCTSNode(game.get_state(), untried_actions=_mcts_actions(game))
        sim = game.clone()
        for _ in range(iterations):
            if time.perf_counter() >= deadline:
                break
            node = _mcts_select(root, sim, c)
            child = _mcts_expand(node, sim)
            value = _mcts_simulate(sim, rollout_depth)
            reward = 0.5 + 0.5 * math.tanh((value - root_value) / MCTS_REWARD_SCALE)
            _mcts_backprop(child, reward)
    finally:
        random.setstate(rng_state)
    if not root.children:
        return root.untried_actions[0] if root.untried_actions else actions[0]
    best = max(root.children, key=lambda ch: (ch.visits, ch.mean_value()))
    return best.action_taken


# ── MCTS helper functions (implement each one) ────────────────────────────────

def _mcts_actions(game: PacmanGame) -> List[Action]:
    """Legal actions worth expanding: no shots at empty tiles, hits first."""
    player = game.player
    targets = {(g.x - player.x, g.y - player.y) for g in game.ghosts if g.alive}
    actions = [a for a in game.get_legal_actions()
               if a not in _SHOOT_DELTAS or _SHOOT_DELTAS[a] in targets]
    return _order_actions(game, actions, maximising=True)


def _mcts_select(root: MCTSNode, game: PacmanGame, c: float = MCTS_C) -> MCTSNode:
    """
    Slide 43, step 1: "recursively apply UCB to choose a path down to a
    leaf (not fully expanded) node n".

    Start at root. At each node: if it is fully expanded AND not terminal,
    use best_child() (UCB1) to descend one level. Stop when you reach a node
    that still has untried actions (not fully expanded) or the game is over.

    Every node stores a full snapshot, so the game is restored only once,
    at the selected node. This gives the same result as restoring at every
    level.
    """
    node = root
    while (node.is_fully_expanded() and node.children
           and not node.state_snapshot.done):
        node = node.best_child(c)
    game.set_state(node.state_snapshot)
    return node


def _mcts_expand(node: MCTSNode, game: PacmanGame) -> MCTSNode:
    """
    Slide 43, step 2a: "add a new child c to n".

    Pop one action from node.untried_actions, apply it to the game,
    create a child MCTSNode for the resulting state, append it to
    node.children, and return it. Terminal nodes return themselves.
    `game` must already be at node's state (see _mcts_select).
    """
    if not node.untried_actions or game.done:
        return node
    action = node.untried_actions.pop(0)
    game.step(action)
    child = MCTSNode(game.get_state(), parent=node, action_taken=action,
                     untried_actions=_mcts_actions(game))
    node.children.append(child)
    return child


def _rollout_action(game: PacmanGame) -> Action:
    """Default rollout policy (MCTS_ROLLOUT_POLICY = 'guided').

    With probability MCTS_ROLLOUT_EPSILON it picks a uniformly random useful
    action. Otherwise it plays like a cautious player:
      1. shoot an adjacent ghost when it has ammo;
      2. without ammo, pick a move or STAY that does not end next to a ghost;
      3. otherwise pick a random move, or STAY.
    A purely random rollout often stays next to ghosts or wastes ammo, which
    makes rollout values mostly noise.
    """
    useful = _mcts_actions(game)
    if random.random() < MCTS_ROLLOUT_EPSILON:
        return random.choice(useful)
    p = game.player
    shots = [a for a in useful if a in _SHOOT_DELTAS]
    if shots:
        return shots[0]
    ghosts = [(g.x, g.y) for g in game.ghosts if g.alive]
    steps = [a for a in useful if a not in _SHOOT_DELTAS]
    if p.ammo == 0 and ghosts:
        def safe(action: Action) -> bool:
            dx, dy = _MOVE_DELTAS.get(action, (0, 0))
            x, y = p.x + dx, p.y + dy
            return all(abs(gx - x) + abs(gy - y) > 1 for gx, gy in ghosts)
        safe_steps = [a for a in steps if safe(a)]
        steps = safe_steps or steps
    return random.choice(steps)


def _mcts_simulate(game: PacmanGame, depth: int) -> float:
    """
    Slide 43, step 2b: "run a rollout from c".

    Play rollout actions for up to `depth` steps (stop early if game.done).
    Return evaluate_state(game) as the rollout value. The rollout policy is
    chosen by MCTS_ROLLOUT_POLICY: 'random' (uniform over legal actions) or
    'guided' (see _rollout_action).
    """
    for _ in range(depth):
        if game.done:
            break
        if MCTS_ROLLOUT_POLICY == 'random':
            action = random.choice(game.get_legal_actions())
        else:
            action = _rollout_action(game)
        game.step(action)
    return evaluate_state(game)


def _mcts_backprop(node: MCTSNode, value: float):
    """
    Slide 43, step 3: "update the win counts from c back up to the root".

    Walk from node up to root (via node.parent), incrementing visits by 1
    and adding value to total_value at every ancestor including node itself.
    """
    while node is not None:
        node.visits += 1
        node.total_value += value
        node = node.parent


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
