# Practical 2 Report — Pacman Agents

## 1. Implementation & Evaluation Heuristic

- **`evaluate_state`**: Assigns terminal priorities (`-1e6` for death, `+1e6` for clearing all ghosts) and combines ghost kills (`+1000`), survival (`+100`), ammo count (`+15`), progress toward the next 10-tile ammo pickup (`+3`), tiles traveled (`+0.5`), and step count (`+0.05`). A single BFS from the player computes wall-aware shortest-path distances to ghosts (replacing raw Manhattan distance where walls block paths), immediate open exits, and reachable tiles. Ghost threat is weighted by behavior (`LOOKAHEAD=1.5`, `FOLLOW/DFS=1.25`, `RANDOM=1.0`), scaled by `1/(d+1)^2`, and reduced when holding ammo for reactive defense. Adjacent shootable ghosts receive an extra bonus when armed.
- **`minimax_agent` & `alphabeta_agent`**: Use iterative deepening within a 4.0s internal deadline (under the 5s limit) on cloned states while restoring RNG state. Alpha-Beta orders moves (adjacent ghost shots first, moves, `STAY`, empty shots last; reversed for MIN) and applies fail-soft pruning, matching Minimax's exact root values while expanding ~10x fewer nodes at depth 4.
- **`mcts_agent` (UCT)**: Implements `ucb1_score`, selection, expansion (pruning empty-tile shots), simulation, and backpropagation with normalized rewards (`0.5 + 0.5 * tanh((v - v_root) / 250)`). Rollouts use an $\epsilon$-greedy guided policy (`rollout_depth=5`, `iterations=500`, `C=1.41`) that shoots adjacent ghosts when armed and avoids tiles adjacent to ghosts when unarmed.

## 2. Results & Algorithm Comparison

Tested across 30 seeded episodes (`10x10` grid, 5 ghosts, `max_steps=100`, 0 timeouts):

| Agent | Settings | Avg Kills | Cleared (5/5) | Survival Rate | Avg Tiles | Avg Steps | Mean / Max Move (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Minimax** | `depth=3` | 2.60 | 20% | 33% | 32.6 | 44.3 | 0.12 / 0.54 |
| **Alpha-Beta** | `depth=4` | 2.53 | 17% | 37% | 29.9 | 46.4 | 0.10 / 0.61 |
| **MCTS (UCT)** | `iter=500, rollout=5, C=1.41` | **4.83** | **93%** | **93%** | **89.3** | **94.8** | 0.19 / 1.40 |

### Why Minimax and Alpha-Beta Underperform
1. **Pessimistic Adversary Model**: In the scaffold's alternating `game.step()` tree, MIN chooses from the player's legal actions to minimize `evaluate_state()`. Thus, every second ply assumes the player intentionally makes its worst possible move (e.g., wasting ammo or stepping into a ghost). In crowded states, all branches lead to forced death under this assumption, causing root scores to tie and deeper searches (`depth >= 6`) to become overly conservative rather than stronger.
2. **Deterministic Single-Sample Horizon**: Each node expansion samples only one random ghost transition, and a 3–4 ply horizon is too short to anticipate multi-ghost encirclement.
3. **Minimax vs. Alpha-Beta**: Alpha-Beta achieves the exact same minimax values as Minimax in a fraction of the nodes, confirming that performance is bounded by the proxy-MIN formulation rather than search depth.

### Why MCTS Excels
MCTS models the actual sequential decision process where the player chooses actions at every step and stochastic ghost reactions are averaged over many simulations. Combining UCB1 exploration with short guided rollouts (`depth=5`) avoids the high variance of purely random rollouts (which survived only 8% in tuning) and consistently clears the board within the 5-second limit.
