# Urban Commute Optimization Report

## Implementation

Neighbors change departure by one time slot or replace a route segment. I used BFS to reconnect to a later waypoint while blocking the retained prefix. Connectivity and the absence of repeated nodes are checked afterward.

HC supports limited sideways moves and random restarts. SA uses Metropolis acceptance and exponential cooling. Beam selects from a shared successor pool. GA uses tournament selection, fitness `1/(1+cost)`, crossover, mutation, and elitism. Function signatures and the simulation engine were kept unchanged.

## Results

The experiments included 575 tuning runs and 400 evaluation runs across five scenarios, using separate seeds. The table shows default settings over ten `rush_hour` runs. Runtime and attainment steps are averages.

| Method | Best / mean travel (min) | Best departure | Runtime (s) | Final-best attainment step |
| --- | ---: | --- | ---: | ---: |
| HC | 41.773 / 41.773 | 08:45 | 0.00391 | 17.4 |
| SA | 41.773 / 41.773 | 08:45 | 0.04975 | 534.5 |
| Beam | 41.773 / 42.170 | 08:45 | 0.00231 | 2.3 |
| GA | 41.773 / 41.773 | 08:45 | 0.18530 | 4.4 |

All four methods found this best route: `Home -> B1 -> B2 -> H2 -> Uni`.

Attainment means the first observation of the final best result, not proven convergence. Step units differ between methods. [Full results](results/summary.csv) cover all five scenarios. Exact small-map solutions were used only for evaluation, never supplied to the solvers.

## Convergence and Parameters

![Convergence comparison](results/convergence_comparison.png)

The figure uses selected settings and seed 42. [Step-based curves](results/convergence_steps.png) are also available. HC makes greedy improvements and can stall on plateaus. SA can escape local minima by accepting worse moves; its [current-cost fluctuations](results/annealing_trajectory.png) differ from its best-so-far curve. Beam stops earlier, while GA retains improvements across generations.

On evaluation seeds, SA with initial temperature 1 was faster, but success fell from 100% to 94%. Beam width 16 raised overall success from 92% to 96%, but performed worse on the plateau scenario. GA retained its baseline settings. I kept the original defaults.

## Analytical Questions

### 1. Gradient Descent

A route is discrete, so ordinary derivatives do not apply. With a fixed route and continuous departure time, differentiable `C_e(t)` allows gradient calculation; a Lipschitz gradient and suitable step size support descent within the allowed interval. Gaussian peaks with `sigma > 0` are smooth, but global optimality is not guaranteed. The result must be mapped back to the time grid and evaluated.

### 2. Local vs. Incremental Search

A complete state makes it easier to change both route and departure and evaluate downstream traffic. Time-dependent Dijkstra is also valid for a fixed departure in a FIFO network. Multiple departure choices, whole-route constraints, or non-FIFO behavior complicate the search. Local search offers approximate results within a limited budget.

### 3. Beam vs. GA

Crossover can combine useful segments from two parents; Beam only selects generated neighbors. With defaults and seed 42, Beam's distinct routes fell from six to two in the dual-peak scenario, without complete collapse. Duplicate states were removed. Route quotas or random immigrants could preserve more diversity.

### 4. Sideways Moves

Equal-length routes with similar traffic can create plateaus. Sideways moves help cross them, but a large allowance can waste time cycling. Reducing the limit from 100 to zero preserved evaluation quality and reduced average runtime across five scenarios from about 45 to 6 ms. This result is specific to these datasets.

### 5. Cooling Schedule

High temperature accepts many poor moves; low temperature risks getting trapped. On tuning seeds, `alpha=0.5` produced a mean optimality gap of 19.16%. With `alpha=0.999`, the gap was zero, but the search reached the 3,000-iteration limit. Fast cooling was cheaper but less accurate.

### 6. Disjoint Parents and Epistasis

Without a shared intermediate node, crossover keeps one parent's route and snaps the average departure to the nearest valid slot. Two good parents can still produce a poor child: changing the first segment or departure changes entry times downstream and may hit a traffic peak.

### 7. State-Space Size

The estimate `N_t b^L` is not a rigorous upper bound when `b` and `L` are averages. For `n=|V|` and maximum out-degree `d`, loop-free routes satisfy:

$$|X| \le N_t\sum_{\ell=1}^{n-1}d^\ell.$$

Exponential growth makes exhaustive search expensive on large networks. Local search explores only part of the space and does not guarantee a global optimum.

## Validation and Tool Use

All 65 tests passed. Reproduce experiments with `python experiments/benchmark.py`.

Codex assisted with code, tests, experiments, and writing. Main instructions, translated from Persian, were: "Complete all requested work while following the PDF" and "Write in a natural student voice."
