# Reproducing the experiments

Run from the repository root with the project dependencies installed:

```powershell
python -m unittest discover -s tests -v
python experiments/benchmark.py
```

The runner uses the frozen `experiments/config.baseline.json`, not the current
default configuration. It writes measurements and figures to `results/` and
never changes solver code or `config.json` automatically. Exact runtimes vary
by machine and system load. Python and library versions, input/source SHA-256
hashes, the source commit, and whether the checkout was dirty during generation
are recorded in `results/metadata.json`.

## Design

- Five compatible map/scenario pairs use the same departure grid, 07:45 through
  08:45 in five-minute increments.
- Each algorithm starts with the same independent RNG seed, regardless of its
  position in a benchmark. Each solver is warmed up before timing.
- Tuning uses seeds 101-105. Each candidate changes one parameter from baseline;
  other parameters remain fixed. Candidates are compared across all five scenarios.
- A configuration is selected for each algorithm using the smallest mean gap
  to the exact reference (rounded to nine decimal places), then the fewest mean
  objective evaluations, then the variant name. Runtime is deliberately excluded
  from selection because it fluctuates between runs.
- Baseline and selected configurations are evaluated on seeds
  0, 1, 2, 3, 4, 5, 6, 7, 8, and 42. These are held-out random seeds, **not** unseen
  maps or traffic scenarios. The comparison does not establish generalization to
  new cities or traffic distributions.
- Measurements are serial. Timings include solver bookkeeping and objective-call
  instrumentation, but exclude reference enumeration, plotting, construction,
  and result validation. One run per seed is timed; reported standard deviations
  include both stochastic search variation and timing noise.
- An external, capped enumerator finds the exact minimum for these small supplied
  maps. It is used only to assess solution quality and never supplies states to
  any solver. It refuses problems with more than 100,000 complete reference states.

## Meaning of the metrics

`cost` is the returned travel duration in minutes. `gap_pct` is
`100 * (cost - reference_cost) / reference_cost`. A trial counts as optimal when
its absolute gap is at most `1e-8` minutes. `optimal_rate` is the fraction of seeds
meeting that criterion.

`steps` is the length of the incumbent history minus one. The unit differs:
HC uses accepted moves and restart starts; SA uses proposals including rejected
ones; Beam uses rounds; GA uses generations. Compare computational effort using
`evaluations` or runtime, not raw steps alone.

`first_best_step` is the first zero-based history index at which that run's final
incumbent appeared. It is an observed attainment step, **not a proof of convergence**
or global optimality. `mean_first_best_step` averages this measurement over seeds.

The evaluation-count curves show the best cost among **all evaluated** candidates;
step curves show each solver's recorded incumbent. An evaluated candidate and an
accepted current state are different concepts. The annealing diagnostic explicitly
shows current-state excursions alongside the best-so-far history.

## Outputs

| File | Contents |
| --- | --- |
| `results/runs.csv` | Every held-out trial, cost, path, departure, runtime, effort, and attainment step |
| `results/summary.csv` | Means, sample standard deviations, success rates, and best routes by profile/scenario/algorithm |
| `results/tuning_runs.csv` | Individual tuning trials |
| `results/tuning_summary.csv` | Parameter comparison pooled across scenarios |
| `results/tuning_by_scenario.csv` | Parameter effects within each scenario |
| `results/config.baseline.json` | Frozen baseline settings used for this run |
| `results/config.tuned.json` | Selected settings, with the original problem definition |
| `results/selection.json` | Winning candidate and its tuning measurements per algorithm |
| `results/references.json` | Exact costs, one optimal path/departure, and enumerated state counts |
| `results/histories.json` | Seed-42 trajectories for baseline and tuned configurations |
| `results/convergence_comparison.png` | Tuned convergence against objective evaluations |
| `results/baseline_convergence.png` | Baseline convergence against objective evaluations |
| `results/convergence_steps.png` | Tuned incumbent histories against solver-specific steps |
| `results/quality_comparison.png` | Baseline vs tuned mean gaps on held-out seeds |
| `results/parameter_sensitivity.png` | Parameter effects on tuning seeds |
| `results/route_diversity.png` | Distinct routes in Beam and GA populations |
| `results/annealing_trajectory.png` | Baseline SA current cost, incumbent, and cooling |
| `results/routes/*.png` | Tuned seed-42 route for every algorithm and scenario |

Route colors represent a traffic **snapshot at departure**, not traffic encountered
at every later edge-entry time. The reported travel duration is still calculated
by the unchanged time-advancing simulation.

## Other runs

```powershell
# One selected configuration through the original CLI:
python main.py --algo all --config results/config.tuned.json --seed 42 --plot-route --plot-metrics --save-figs

# An independent experiment with other, non-overlapping random seeds:
python experiments/benchmark.py --seeds 11,12,13 --tuning-seeds 201,202,203 --output results-alternative
```

The supplied Gaussian edge-time model does not automatically guarantee the
mathematical FIFO condition `1 + d(tau)/dt >= 0`. All experiments use the provided
evaluator unchanged; no result assumes an unverified FIFO property.
