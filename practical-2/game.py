"""
game.py — Pacman-like game engine.

Importable API:
    from game import PacmanGame, GameConfig, Action

Quick-start (AI):
    game = PacmanGame(config=GameConfig(num_ghosts=4))
    obs  = game.reset()
    obs, reward, done, info = game.step(Action.RIGHT)

Quick-start (human window):
    game = PacmanGame(render=True)
    game.run()

Controls (human play):
    Arrow keys / WASD  → move
    Shift + direction  → shoot in that direction
    U / I / J / K      → shoot diagonally (up-left / up-right / down-left / down-right)
    Numpad 7 / 9 / 1 / 3 also shoot diagonally
    R                  → reset
    Q                  → quit
"""

import copy
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import pygame


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GameConfig:
    grid_width:      int   = 10    # columns
    grid_height:     int   = 10    # rows
    num_ghosts:      int   = 5
    tiles_per_ammo:  int   = 10    # player gains 1 ammo every N tiles walked
    cell_size:       int   = 54    # pixels per tile
    fps:             int   = 10
    diagonal_shooting: bool = True  # allow SHOOT_UP_LEFT / UP_RIGHT / DOWN_LEFT / DOWN_RIGHT
    max_steps:       int   = 100
    # ── maze obstacles ────────────────────────────────────────────────────────
    # The board is laid out as a lattice of "rooms" (corridor_width wide)
    # separated by walls (wall_thickness thick). A randomized DFS carves a
    # perfect maze through the lattice, then the knobs below open it up.
    maze_corridor_width: int   = 1     # tiles per corridor
    maze_wall_thickness: int   = 1     # tiles per wall
    maze_straightness:   float = 0.3   # 0 = twisty, 1 = long straight runs
    maze_braid:          float = 0.5   # chance each dead end gets opened into a loop
    maze_loop_chance:    float = 0.1   # chance any remaining inner wall is knocked out
    maze_pillar_removal: float = 0.0   # chance a lone wall pillar (all 4 sides open) is removed
    maze_symmetric:      bool  = False # mirror left/right like classic Pac-Man
    maze_seed:  Optional[int]  = None  # fixed seed → same maze every reset
    obstacle_clear_radius: int = 1     # wall-free zone around player spawn


# ──────────────────────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────────────────────

class Action(Enum):
    UP          = 0
    DOWN        = 1
    LEFT        = 2
    RIGHT       = 3
    SHOOT_UP    = 4
    SHOOT_DOWN  = 5
    SHOOT_LEFT  = 6
    SHOOT_RIGHT = 7
    SHOOT_UP_LEFT    = 8
    SHOOT_UP_RIGHT   = 9
    SHOOT_DOWN_LEFT  = 10
    SHOOT_DOWN_RIGHT = 11
    STAY         = 12   # player skips turn; ghosts still move


_MOVE_DELTA = {
    Action.UP:    ( 0, -1),
    Action.DOWN:  ( 0,  1),
    Action.LEFT:  (-1,  0),
    Action.RIGHT: ( 1,  0),
}
_SHOOT_DELTA = {
    Action.SHOOT_UP:    ( 0, -1),
    Action.SHOOT_DOWN:  ( 0,  1),
    Action.SHOOT_LEFT:  (-1,  0),
    Action.SHOOT_RIGHT: ( 1,  0),
    Action.SHOOT_UP_LEFT:    (-1, -1),
    Action.SHOOT_UP_RIGHT:   ( 1, -1),
    Action.SHOOT_DOWN_LEFT:  (-1,  1),
    Action.SHOOT_DOWN_RIGHT: ( 1,  1),
}
_DIAGONAL_SHOOTS = {a for a, (dx, dy) in _SHOOT_DELTA.items() if dx and dy}


# ──────────────────────────────────────────────────────────────────────────────
# Ghost behaviour
# ──────────────────────────────────────────────────────────────────────────────

class GhostBehavior(Enum):
    RANDOM    = auto()  # move to a random adjacent tile
    FOLLOW    = auto()  # greedy Manhattan-distance minimiser
    LOOKAHEAD = auto()  # depth-limited search (default depth = 5)
    DFS       = auto()  # depth-first search path to player


_LOOKAHEAD_DEPTH = 5


# ──────────────────────────────────────────────────────────────────────────────
# Ghost
# ──────────────────────────────────────────────────────────────────────────────

class Ghost:
    def __init__(self, x: int, y: int,
                 behavior: Optional[GhostBehavior] = None):
        self.x        = x
        self.y        = y
        _chasing = [GhostBehavior.FOLLOW, GhostBehavior.LOOKAHEAD, GhostBehavior.DFS]
        self.behavior = behavior or random.choice(_chasing)
        self.alive    = True

    # ── movement helpers (return desired position, do NOT mutate self) ────────

    def _neighbors(self, x: int, y: int, w: int, h: int, walls: set):
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in walls:
                yield nx, ny

    def _desired_random(self, cfg: GameConfig, walls: set) -> Tuple[int, int]:
        choices = list(self._neighbors(self.x, self.y,
                                       cfg.grid_width, cfg.grid_height, walls))
        return random.choice(choices) if choices else (self.x, self.y)

    def _bfs_first_step(self, tx: int, ty: int, cfg: GameConfig,
                        walls: set, shuffle: bool = False) -> Tuple[int, int]:
        """BFS shortest path to (tx, ty); return first step. Wall-aware."""
        from collections import deque
        if (self.x, self.y) == (tx, ty):
            return (self.x, self.y)
        w, h = cfg.grid_width, cfg.grid_height
        queue: deque = deque([(self.x, self.y)])
        parent: dict = {(self.x, self.y): None}
        while queue:
            cx, cy = queue.popleft()
            nbrs = list(self._neighbors(cx, cy, w, h, walls))
            if shuffle:
                random.shuffle(nbrs)
            for nx, ny in nbrs:
                if (nx, ny) not in parent:
                    parent[(nx, ny)] = (cx, cy)
                    if nx == tx and ny == ty:
                        node = (tx, ty)
                        while parent[node] != (self.x, self.y):
                            node = parent[node]
                        return node
                    queue.append((nx, ny))
        return self._desired_random(cfg, walls)  # unreachable

    def _desired_follow(self, px: int, py: int,
                        cfg: GameConfig, walls: set) -> Tuple[int, int]:
        return self._bfs_first_step(px, py, cfg, walls)

    def _desired_lookahead(self, px: int, py: int, cfg: GameConfig,
                           walls: set,
                           depth: int = _LOOKAHEAD_DEPTH) -> Tuple[int, int]:
        """BFS toward intercepted position: target a tile depth steps ahead of
        player along the ghost→player axis to cut off escape routes."""
        w, h = cfg.grid_width, cfg.grid_height
        dx = 1 if px > self.x else (-1 if px < self.x else 0)
        dy = 1 if py > self.y else (-1 if py < self.y else 0)
        # intercept point clamped to grid
        tx = max(0, min(w - 1, px + dx * depth))
        ty = max(0, min(h - 1, py + dy * depth))
        # if intercept is a wall, fall back to direct chase
        if (tx, ty) in walls:
            tx, ty = px, py
        return self._bfs_first_step(tx, ty, cfg, walls)

    def _desired_dfs(self, px: int, py: int,
                     cfg: GameConfig, walls: set) -> Tuple[int, int]:
        """BFS with shuffled neighbor order — finds shortest path but varies
        which of the equally-short paths it takes, creating flank diversity."""
        return self._bfs_first_step(px, py, cfg, walls, shuffle=True)

    def compute_move(self, px: int, py: int,
                     cfg: GameConfig, walls: set) -> Tuple[int, int]:
        """Return desired (x, y) without mutating state."""
        if not self.alive:
            return (self.x, self.y)
        if self.behavior == GhostBehavior.RANDOM:
            return self._desired_random(cfg, walls)
        elif self.behavior == GhostBehavior.FOLLOW:
            return self._desired_follow(px, py, cfg, walls)
        elif self.behavior == GhostBehavior.LOOKAHEAD:
            return self._desired_lookahead(px, py, cfg, walls)
        elif self.behavior == GhostBehavior.DFS:
            return self._desired_dfs(px, py, cfg, walls)
        return (self.x, self.y)


# ──────────────────────────────────────────────────────────────────────────────
# Player
# ──────────────────────────────────────────────────────────────────────────────

class Player:
    def __init__(self, x: int, y: int):
        self.x              = x
        self.y              = y
        self.ammo           = 1   # starts with one shot
        self.tiles_traveled = 0
        self.alive          = True


# ──────────────────────────────────────────────────────────────────────────────
# Serialisable game-state snapshot (for tree search / cloning)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    player_x:              int
    player_y:              int
    player_ammo:           int
    player_tiles_traveled: int
    player_alive:          bool
    ghosts:                List[dict]  # {x, y, behavior, alive}
    step_count:            int
    score:                 float
    done:                  bool
    walls:                 List[Tuple[int, int]]  # obstacle tiles


# ──────────────────────────────────────────────────────────────────────────────
# Main game class
# ──────────────────────────────────────────────────────────────────────────────

class PacmanGame:
    """
    Pacman-like game engine — importable from any Python file.

    Observation dict (returned by reset / step):
        player_x, player_y  : int  — player position
        player_ammo         : int  — shots remaining
        player_alive        : bool
        tiles_traveled      : int
        ghosts              : list of {x, y, alive, behavior}
        step                : int
        grid_width          : int
        grid_height         : int

    Actions: see the Action enum (UP/DOWN/LEFT/RIGHT/SHOOT_UP/…, plus
    diagonal SHOOT_UP_LEFT/… when config.diagonal_shooting is on).

    Reward is intentionally minimal here — students design the evaluation
    function in student_assignment.py.
    """

    _COLORS = dict(
        bg     = ( 20,  20,  20),
        grid   = ( 40,  40,  40),
        wall   = ( 80, 120, 160),
        player = (255, 220,   0),
        ghost  = (220,  30,  30),
        text   = (255, 255, 255),
        dead   = ( 80,  80,  80),
    )

    def __init__(self, config: Optional[GameConfig] = None,
                 render: bool = False):
        self.config          = config or GameConfig()
        self._render_enabled = render
        self._screen         = None
        self._clock          = None
        # these are set by reset()
        self.player:    Player      = None  # type: ignore[assignment]
        self.ghosts:    List[Ghost] = []
        self.walls:     set         = set()
        self.step_count:    int        = 0
        self.score:         float      = 0.0
        self.done:          bool       = False
        self.agent_timeouts: int       = 0
        self.reset()

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> Dict:
        cfg = self.config
        cx, cy = cfg.grid_width // 2, cfg.grid_height // 2
        self.player = Player(cx, cy)
        self.walls  = self._generate_walls(cfg, cx, cy)
        self.ghosts = []
        occupied    = {(cx, cy)} | self.walls

        for _ in range(cfg.num_ghosts):
            for _ in range(10_000):
                gx = random.randint(0, cfg.grid_width  - 1)
                gy = random.randint(0, cfg.grid_height - 1)
                too_close = abs(gx - cx) + abs(gy - cy) <= 5
                if (gx, gy) not in occupied and not too_close:
                    occupied.add((gx, gy))
                    self.ghosts.append(Ghost(gx, gy))
                    break

        self.step_count = 0
        self.score      = 0.0
        self.done       = False
        return self._obs()

    @staticmethod
    def _generate_walls(cfg: 'GameConfig', cx: int, cy: int) -> set:
        rng = random.Random(cfg.maze_seed) if cfg.maze_seed is not None else random
        W, H  = cfg.grid_width, cfg.grid_height
        cw    = max(1, cfg.maze_corridor_width)
        wt    = max(1, cfg.maze_wall_thickness)
        pitch = cw + wt
        cols  = max(1, (W + wt) // pitch)
        rows  = max(1, (H + wt) // pitch)
        # centre the lattice; leftover tiles form an open outer corridor
        ox = (W - (cols * pitch - wt)) // 2
        oy = (H - (rows * pitch - wt)) // 2

        def _edge(a, b):
            return (a, b) if a <= b else (b, a)

        def _nbrs(c, r):
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = c + dc, r + dr
                if 0 <= nc < cols and 0 <= nr < rows:
                    yield (nc, nr), (dc, dr)

        all_edges = {_edge((c, r), n) for c in range(cols) for r in range(rows)
                     for n, _ in _nbrs(c, r)}

        # 1. perfect maze via randomized DFS (with straightness bias)
        passages = set()
        start    = (rng.randrange(cols), rng.randrange(rows))
        visited  = {start}
        stack    = [(start, None)]
        while stack:
            cell, last_dir = stack[-1]
            options = [(n, d) for n, d in _nbrs(*cell) if n not in visited]
            if not options:
                stack.pop()
                continue
            straight = [o for o in options if o[1] == last_dir]
            if straight and rng.random() < cfg.maze_straightness:
                nxt, d = straight[0]
            else:
                nxt, d = rng.choice(options)
            passages.add(_edge(cell, nxt))
            visited.add(nxt)
            stack.append((nxt, d))

        # 2. braid: open dead ends (prefer joining two dead ends together)
        def _degree(cell):
            return sum(1 for n, _ in _nbrs(*cell) if _edge(cell, n) in passages)

        cells = [(c, r) for c in range(cols) for r in range(rows)]
        rng.shuffle(cells)
        for cell in cells:
            if _degree(cell) != 1 or rng.random() >= cfg.maze_braid:
                continue
            closed = [n for n, _ in _nbrs(*cell) if _edge(cell, n) not in passages]
            if closed:
                dead = [n for n in closed if _degree(n) == 1]
                passages.add(_edge(cell, rng.choice(dead or closed)))

        # 3. extra loops: knock out random remaining walls
        for e in sorted(all_edges - passages):
            if rng.random() < cfg.maze_loop_chance:
                passages.add(e)

        # 4. optional left/right mirror
        def _mirror(e):
            (c1, r1), (c2, r2) = e
            return _edge((cols - 1 - c1, r1), (cols - 1 - c2, r2))

        if cfg.maze_symmetric:
            left = {e for e in passages if e[0][0] + e[1][0] <= cols - 1}
            passages = left | {_mirror(e) for e in left}

        # 5. guarantee every cell is reachable (union-find over the lattice)
        parent = {cell: cell for cell in cells}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in passages:
            parent[_find(a)] = _find(b)
        candidates = sorted(all_edges - passages)
        rng.shuffle(candidates)
        for e in candidates:
            group = [e, _mirror(e)] if cfg.maze_symmetric else [e]
            if _find(e[0]) != _find(e[1]):
                for a, b in group:
                    passages.add(_edge(a, b))
                    parent[_find(a)] = _find(b)

        # 6. rasterise: start with the whole lattice walled, carve rooms + passages
        def _rect(x0, y0, w, h):
            return {(x, y) for x in range(x0, x0 + w) for y in range(y0, y0 + h)
                    if 0 <= x < W and 0 <= y < H}

        walls = _rect(ox, oy, cols * pitch - wt, rows * pitch - wt)
        for c, r in cells:
            walls -= _rect(ox + c * pitch, oy + r * pitch, cw, cw)
        for (c1, r1), (c2, r2) in passages:
            if c1 != c2:   # horizontal neighbours → open the vertical wall between
                walls -= _rect(ox + c1 * pitch + cw, oy + r1 * pitch, wt, cw)
            else:          # vertical neighbours → open the horizontal wall between
                walls -= _rect(ox + c1 * pitch, oy + r1 * pitch + cw, cw, wt)

        # 7. remove isolated pillars (wall blocks with all four sides open)
        for c in range(cols - 1):
            for r in range(rows - 1):
                ring = [_edge((c, r), (c + 1, r)), _edge((c, r + 1), (c + 1, r + 1)),
                        _edge((c, r), (c, r + 1)), _edge((c + 1, r), (c + 1, r + 1))]
                if (all(e in passages for e in ring)
                        and rng.random() < cfg.maze_pillar_removal):
                    walls -= _rect(ox + c * pitch + cw, oy + r * pitch + cw, wt, wt)

        # 8. clear zone around player spawn (spawn tile itself is always open)
        clear_r = cfg.obstacle_clear_radius
        walls = {(x, y) for x, y in walls
                 if abs(x - cx) > clear_r or abs(y - cy) > clear_r}
        walls.discard((cx, cy))

        # 9. safety net: carve through walls to any open pocket the spawn can't reach
        while True:
            seen, frontier = {(cx, cy)}, [(cx, cy)]
            while frontier:
                x, y = frontier.pop()
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    n = (x + dx, y + dy)
                    if (0 <= n[0] < W and 0 <= n[1] < H
                            and n not in walls and n not in seen):
                        seen.add(n)
                        frontier.append(n)
            if len(seen) + len(walls) == W * H:
                break
            # BFS outward from the reachable region through walls to a pocket
            prev  = {s: None for s in seen}
            queue = list(seen)
            hit   = None
            for x, y in queue:
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    n = (x + dx, y + dy)
                    if 0 <= n[0] < W and 0 <= n[1] < H and n not in prev:
                        prev[n] = (x, y)
                        if n not in walls:
                            hit = n
                            break
                        queue.append(n)
                if hit:
                    break
            while hit is not None:
                walls.discard(hit)
                hit = prev[hit]

        return walls

    def step(self, action: Action) -> Tuple[Dict, float, bool, Dict]:
        """
        Apply action, advance ghosts, check collisions.

        Returns (observation, reward, done, info).

        Reward values are intentional placeholders — students should override
        evaluate_state() in student_assignment.py to design their own.
        """
        if self.done:
            return self._obs(), 0.0, True, self._info()

        cfg    = self.config
        p      = self.player
        reward = 0.0

        # ── player move or shoot ──────────────────────────────────────────────
        if action in _MOVE_DELTA:
            dx, dy = _MOVE_DELTA[action]
            nx, ny = p.x + dx, p.y + dy
            if (0 <= nx < cfg.grid_width and 0 <= ny < cfg.grid_height
                    and (nx, ny) not in self.walls):
                p.x, p.y = nx, ny
                p.tiles_traveled += 1
                if p.tiles_traveled % cfg.tiles_per_ammo == 0:
                    p.ammo += 1

        elif (action in _SHOOT_DELTA and p.ammo > 0
              and (cfg.diagonal_shooting or action not in _DIAGONAL_SHOOTS)):
            dx, dy = _SHOOT_DELTA[action]
            tx, ty = p.x + dx, p.y + dy
            hit = False
            for g in self.ghosts:
                if g.alive and g.x == tx and g.y == ty:
                    g.alive = False
                    reward += 10.0
                    hit = True
            p.ammo -= 1              # ammo spent regardless of hit
            if hit:                  # slide onto the cleared tile
                p.x, p.y = tx, ty
                p.tiles_traveled += 1
                if p.tiles_traveled % cfg.tiles_per_ammo == 0:
                    p.ammo += 1

        # ── ghost moves (simultaneous; no two ghosts share a tile) ──────────
        # 1. All ghosts decide based on current positions (no sequential bias).
        alive_ghosts = [g for g in self.ghosts if g.alive]
        proposals    = {g: g.compute_move(p.x, p.y, cfg, self.walls)
                        for g in alive_ghosts}
        # 2. Resolve conflicts: shuffle for fairness, then claim tiles greedily.
        order = list(alive_ghosts)
        random.shuffle(order)
        occupied = set()
        for g in order:
            dest = proposals[g]
            if dest not in occupied:
                g.x, g.y = dest
            # if contested: ghost stays (current tile re-added below)
            occupied.add((g.x, g.y))

        # ── collision check (reactive defence: ammo kills ghost on entry) ───────
        for g in self.ghosts:
            if g.alive and g.x == p.x and g.y == p.y:
                if p.ammo > 0:
                    g.alive  = False
                    p.ammo  -= 1
                    reward  += 10.0
                else:
                    p.alive  = False

        self.step_count += 1
        self.score      += reward

        if not p.alive or self.step_count >= cfg.max_steps:
            self.done = True
            if not p.alive:
                reward -= 100.0  # placeholder death penalty

        return self._obs(), reward, self.done, self._info()

    def get_legal_actions(self) -> List[Action]:
        """Return all actions the player can legally take this turn."""
        p   = self.player
        cfg = self.config
        actions = []
        for a, (dx, dy) in _MOVE_DELTA.items():
            nx, ny = p.x + dx, p.y + dy
            if (0 <= nx < cfg.grid_width and 0 <= ny < cfg.grid_height
                    and (nx, ny) not in self.walls):
                actions.append(a)
        if p.ammo > 0:
            actions.extend(a for a in _SHOOT_DELTA
                           if cfg.diagonal_shooting or a not in _DIAGONAL_SHOOTS)
        actions.append(Action.STAY)
        return actions

    # ── state serialisation (for planning algorithms) ─────────────────────────

    def get_state(self) -> GameState:
        """Snapshot the entire game state."""
        p = self.player
        return GameState(
            player_x              = p.x,
            player_y              = p.y,
            player_ammo           = p.ammo,
            player_tiles_traveled = p.tiles_traveled,
            player_alive          = p.alive,
            ghosts                = [
                {'x': g.x, 'y': g.y, 'behavior': g.behavior, 'alive': g.alive}
                for g in self.ghosts
            ],
            step_count = self.step_count,
            score      = self.score,
            done       = self.done,
            walls      = list(self.walls),
        )

    def set_state(self, state: GameState):
        """Restore game to a previously saved snapshot."""
        self.player                   = Player(state.player_x, state.player_y)
        self.player.ammo              = state.player_ammo
        self.player.tiles_traveled    = state.player_tiles_traveled
        self.player.alive             = state.player_alive
        self.ghosts = []
        for gd in state.ghosts:
            g = Ghost(gd['x'], gd['y'], gd['behavior'])
            g.alive = gd['alive']
            self.ghosts.append(g)
        self.step_count = state.step_count
        self.score      = state.score
        self.done       = state.done
        self.walls      = set(map(tuple, state.walls))

    def clone(self) -> 'PacmanGame':
        """Return an independent copy of the game (no renderer) for simulation."""
        # bypass __init__ so no maze is generated just to be thrown away
        c = PacmanGame.__new__(PacmanGame)
        c.config          = copy.copy(self.config)
        c._render_enabled = False
        c._screen = c._clock = c._font = c._freetype_font = None
        c.player = Player(self.player.x, self.player.y)
        c.player.ammo           = self.player.ammo
        c.player.tiles_traveled = self.player.tiles_traveled
        c.player.alive          = self.player.alive
        c.ghosts = []
        for g in self.ghosts:
            ng = Ghost(g.x, g.y, g.behavior)
            ng.alive = g.alive
            c.ghosts.append(ng)
        c.walls      = self.walls   # never mutated after reset(), safe to share
        c.step_count = self.step_count
        c.score      = self.score
        c.done       = self.done
        return c

    # ── rendering / human play ────────────────────────────────────────────────

    def render(self):
        """Draw one frame. Call after each step when rendering is desired."""
        self._init_pygame()
        cfg = self.config
        cs  = cfg.cell_size
        scr = self._screen
        C   = self._COLORS

        scr.fill(C['bg'])

        for x in range(cfg.grid_width + 1):
            pygame.draw.line(scr, C['grid'], (x * cs, 0),
                             (x * cs, cfg.grid_height * cs))
        for y in range(cfg.grid_height + 1):
            pygame.draw.line(scr, C['grid'], (0, y * cs),
                             (cfg.grid_width * cs, y * cs))

        for wx, wy in self.walls:
            pygame.draw.rect(scr, C['wall'],
                             (wx * cs + 1, wy * cs + 1, cs - 2, cs - 2))

        for g in self.ghosts:
            if g.alive:
                pygame.draw.circle(scr, C['ghost'],
                                   (g.x * cs + cs // 2, g.y * cs + cs // 2),
                                   cs // 2 - 1)

        p   = self.player
        col = C['player'] if p.alive else C['dead']
        pygame.draw.circle(scr, col,
                           (p.x * cs + cs // 2, p.y * cs + cs // 2),
                           cs // 2 - 1)

        pygame.display.flip()
        print(f"Step:{self.step_count}  Score:{self.score:.0f}  Ammo:{p.ammo}  "
              f"Tiles:{p.tiles_traveled}  Ghosts:{sum(1 for g in self.ghosts if g.alive)}  "
              f"Timeouts:{self.agent_timeouts}", end='\r')

    def run(self, agent_fn=None):
        """
        Open a pygame window and play.

        agent_fn: callable(obs) -> Action, or None for human keyboard control.

        Human controls:
            Arrow keys / WASD           — move
            Shift + Arrow key / WASD    — shoot in that direction
            U / I / J / K               — shoot up-left / up-right / down-left / down-right
            Numpad 7 / 9 / 1 / 3        — same diagonals
            R                           — reset
            Q                           — quit
        """
        self._init_pygame()
        obs = self.reset()

        _MOVE_KEYS = {
            pygame.K_UP:    Action.UP,    pygame.K_w: Action.UP,
            pygame.K_DOWN:  Action.DOWN,  pygame.K_s: Action.DOWN,
            pygame.K_LEFT:  Action.LEFT,  pygame.K_a: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT, pygame.K_d: Action.RIGHT,
        }
        _SHOOT_KEYS = {
            pygame.K_UP:    Action.SHOOT_UP,    pygame.K_w: Action.SHOOT_UP,
            pygame.K_DOWN:  Action.SHOOT_DOWN,  pygame.K_s: Action.SHOOT_DOWN,
            pygame.K_LEFT:  Action.SHOOT_LEFT,  pygame.K_a: Action.SHOOT_LEFT,
            pygame.K_RIGHT: Action.SHOOT_RIGHT, pygame.K_d: Action.SHOOT_RIGHT,
        }
        _DIAG_KEYS = {
            pygame.K_u: Action.SHOOT_UP_LEFT,    pygame.K_KP7: Action.SHOOT_UP_LEFT,
            pygame.K_i: Action.SHOOT_UP_RIGHT,   pygame.K_KP9: Action.SHOOT_UP_RIGHT,
            pygame.K_j: Action.SHOOT_DOWN_LEFT,  pygame.K_KP1: Action.SHOOT_DOWN_LEFT,
            pygame.K_k: Action.SHOOT_DOWN_RIGHT, pygame.K_KP3: Action.SHOOT_DOWN_RIGHT,
        }

        running = True
        while running:
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        obs = self.reset()
                    elif agent_fn is None:
                        mods = pygame.key.get_mods()
                        if event.key in _DIAG_KEYS:
                            action = _DIAG_KEYS[event.key]
                        elif mods & pygame.KMOD_SHIFT and event.key in _SHOOT_KEYS:
                            action = _SHOOT_KEYS[event.key]
                        elif event.key in _MOVE_KEYS:
                            action = _MOVE_KEYS[event.key]

            if agent_fn is not None and not self.done:
                action = agent_fn(obs)

            if action is not None and not self.done:
                obs, _, _, _ = self.step(action)

            self.render()
            self._clock.tick(self.config.fps)

        pygame.quit()

    # ── internals ─────────────────────────────────────────────────────────────

    def _obs(self) -> Dict:
        p = self.player
        return {
            'player_x':       p.x,
            'player_y':       p.y,
            'player_ammo':    p.ammo,
            'player_alive':   p.alive,
            'tiles_traveled': p.tiles_traveled,
            'ghosts': [
                {'x': g.x, 'y': g.y, 'alive': g.alive,
                 'behavior': g.behavior.name}
                for g in self.ghosts
            ],
            'walls':       list(self.walls),
            'step':        self.step_count,
            'grid_width':  self.config.grid_width,
            'grid_height': self.config.grid_height,
        }

    def _info(self) -> Dict:
        return {
            'score':        self.score,
            'ammo':         self.player.ammo,
            'tiles_traveled': self.player.tiles_traveled,
            'step':         self.step_count,
            'ghosts_alive': sum(1 for g in self.ghosts if g.alive),
        }

    def _init_pygame(self):
        if self._screen is not None:
            return
        pygame.init()
        cfg = self.config
        w   = cfg.grid_width  * cfg.cell_size
        h   = cfg.grid_height * cfg.cell_size
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption('PacmanGame')
        self._clock = pygame.time.Clock()


# ──────────────────────────────────────────────────────────────────────────────
# Script entry point — human play
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Controls: Arrow keys / WASD to move, Shift+direction to shoot, "
          "U/I/J/K or numpad 7/9/1/3 to shoot diagonally, R to reset, Q to quit")
    PacmanGame(render=True).run()
