"""Reproducible tuning and held-out benchmarks; core simulation is unchanged."""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import SOLVER_REGISTRY, build_problem


SCENARIOS = {
    "rush_hour": "toy_graph",
    "dual_peak_rush_hour": "metropolis_graph",
    "cascading_gridlock": "metropolis_graph",
    "crossover_showcase": "metropolis_graph",
    "flat_plateau": "downtown_grid",
}
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
TUNING_SEEDS = [101, 102, 103, 104, 105]


class CountedProblem:
    """Count solver calls to evaluate; record best *evaluated* cost vs effort."""

    def __init__(self, problem):
        self.problem = problem
        self.evaluations = 0
        self.trace = []
        self.best = float("inf")

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def evaluate(self, state):
        value = self.problem.evaluate(state)
        self.evaluations += 1
        if value < self.best:
            self.best = value
            self.trace.append([self.evaluations, value])
        return value


def make_problem(scenario, config):
    return build_problem(
        str(ROOT / "data" / "maps" / f"{SCENARIOS[scenario]}.json"),
        str(ROOT / "data" / "scenarios" / f"{scenario}.json"), config["problem"],
    )


def exact_reference(problem, max_states=100000):
    """Enumerate small supplied maps only; never used by any search solver."""
    stack = [(problem.origin,)]
    paths = []
    times = problem.get_valid_time_domain()
    while stack:
        path = stack.pop()
        if path[-1] == problem.destination:
            paths.append(path)
            if len(paths) * len(times) > max_states:
                raise ValueError("Reference state budget exceeded; do not enumerate large maps.")
        else:
            for node in problem.network.get_neighbors(path[-1]):
                if node not in path:
                    stack.append(path + (node,))
    if not paths:
        raise RuntimeError("No valid route for exact reference.")
    from core.state import State
    states = (State(path, departure) for path in paths for departure in times)
    best = min(states, key=problem.evaluate)
    return {"cost": problem.evaluate(best), "departure": best.departure_time,
            "path": list(best.path), "paths": len(paths), "time_slots": len(times),
            "states": len(paths) * len(times)}


def trial(problem, algo, params, seed, reference):
    random.seed(seed)  # Reset immediately before each solver, independent of order.
    counted = CountedProblem(problem)
    solver = SOLVER_REGISTRY[algo][1](counted, **params)
    start = time.perf_counter()
    state, cost = solver.solve()
    elapsed = time.perf_counter() - start
    if (not problem.is_valid_state(state)
            or state.departure_time not in problem.get_valid_time_domain()
            or not math.isfinite(cost)
            or not math.isclose(cost, problem.evaluate(state), rel_tol=0, abs_tol=1e-8)):
        raise AssertionError(f"Invalid result from {algo}, seed {seed}.")
    if cost < reference["cost"] - 1e-8:
        raise AssertionError("Search result contradicts exact reference.")
    if not solver.history or solver.history[-1] != cost:
        raise AssertionError("Incumbent history does not end at returned cost.")
    if any(a < b for a, b in zip(solver.history, solver.history[1:])):
        raise AssertionError("Incumbent history increased.")
    first_best = next(i for i, value in enumerate(solver.history) if value == cost)
    row = {"algo": algo, "seed": seed, "cost": cost,
           "gap_pct": 100 * (cost - reference["cost"]) / max(reference["cost"], 1e-12),
           "optimal": abs(cost - reference["cost"]) <= 1e-8,
           "runtime_s": elapsed, "evaluations": counted.evaluations,
           "steps": len(solver.history) - 1, "first_best_step": first_best,
           "departure": state.departure_time, "departure_hhmm": state.format_time(),
           "path": " -> ".join(state.path)}
    if counted.trace and counted.trace[-1][0] != counted.evaluations:
        counted.trace.append([counted.evaluations, counted.best])
    history = {"seed": seed, "best": solver.history,
               "current": solver.current_history, "evaluation_trace": counted.trace,
               "temperature": getattr(solver, "temperature_history", []),
               "path_diversity": getattr(solver, "path_diversity_history", []),
               "restarts": getattr(solver, "restart_steps", []),
               "path": list(state.path), "departure": state.departure_time, "cost": cost}
    return row, history


def summarize(rows, group_keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    result = []
    for keys, group in groups.items():
        best = min(group, key=lambda item: (item["cost"], item["seed"]))
        summary = dict(zip(group_keys, keys))
        summary.update({"runs": len(group), "best_cost": best["cost"],
                        "mean_cost": statistics.mean(r["cost"] for r in group),
                        "std_cost": statistics.stdev(r["cost"] for r in group) if len(group) > 1 else 0,
                        "mean_gap_pct": statistics.mean(r["gap_pct"] for r in group),
                        "optimal_rate": statistics.mean(int(r["optimal"]) for r in group),
                        "mean_runtime_s": statistics.mean(r["runtime_s"] for r in group),
                        "std_runtime_s": statistics.stdev(r["runtime_s"] for r in group) if len(group) > 1 else 0,
                        "mean_evaluations": statistics.mean(r["evaluations"] for r in group),
                        "mean_steps": statistics.mean(r["steps"] for r in group),
                        "mean_first_best_step": statistics.mean(r["first_best_step"] for r in group),
                        "best_seed": best["seed"], "best_departure": best["departure_hhmm"],
                        "best_path": best["path"]})
        result.append(summary)
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def variants(config, algo):
    base = config[SOLVER_REGISTRY[algo][2]]
    changes = {
        "hc": [("max_sideways", 0), ("max_sideways", 10),
               ("max_restarts", 0), ("max_restarts", 2)],
        "sa": [("initial_temp", 1.0), ("initial_temp", 1000.0),
               ("cooling_rate", 0.5), ("cooling_rate", 0.9), ("cooling_rate", 0.999)],
        "beam": [("beam_width", 1), ("beam_width", 3), ("beam_width", 16)],
        "ga": [("mutation_rate", 0.0), ("mutation_rate", 0.1), ("mutation_rate", 0.5),
               ("elitism", 0), ("elitism", 1), ("population_size", 20),
               ("population_size", 100)],
    }
    yield "baseline", deepcopy(base)
    for key, value in changes[algo]:
        yield f"{key}={value}", {**base, key: value}


def run(config, seeds, tuning_seeds, out):
    if set(seeds) & set(tuning_seeds):
        raise ValueError("Evaluation and tuning seeds must be disjoint.")
    out.mkdir(parents=True, exist_ok=True)
    problems = {scenario: make_problem(scenario, config) for scenario in SCENARIOS}
    references = {name: exact_reference(problem) for name, problem in problems.items()}
    write_json(out / "references.json", references)

    # Warm imports and code paths before timed comparisons. Nothing is cached
    # between solvers, and plot generation is outside all measured intervals.
    for algo, (_, _, key) in SOLVER_REGISTRY.items():
        trial(problems["rush_hour"], algo, config[key], 9999, references["rush_hour"])

    tuning_rows = []
    variant_configs = {}
    for algo in SOLVER_REGISTRY:
        for label, params in variants(config, algo):
            variant_configs[(algo, label)] = params
            for scenario, problem in problems.items():
                for seed in tuning_seeds:
                    row, _ = trial(problem, algo, params, seed, references[scenario])
                    tuning_rows.append({"scenario": scenario, "variant": label, **row})
            print(f"Tuned {algo}: {label}", flush=True)
    write_csv(out / "tuning_runs.csv", tuning_rows)
    tuning_summary = summarize(tuning_rows, ["algo", "variant"])
    write_csv(out / "tuning_summary.csv", tuning_summary)
    write_csv(out / "tuning_by_scenario.csv", summarize(tuning_rows, ["scenario", "algo", "variant"]))

    tuned = deepcopy(config)
    selection = {}
    for algo, (_, _, key) in SOLVER_REGISTRY.items():
        candidates = [row for row in tuning_summary if row["algo"] == algo]
        # Quality first; round only negligible gap noise, then prefer fewer
        # objective calls. This makes selection independent of timing jitter.
        winner = min(candidates, key=lambda row: (round(row["mean_gap_pct"], 9),
                                                  row["mean_evaluations"], row["variant"]))
        tuned[key] = variant_configs[(algo, winner["variant"])]
        selection[algo] = winner
    write_json(out / "config.tuned.json", tuned)
    write_json(out / "selection.json", selection)

    rows = []
    histories = {}
    representative_seed = 42 if 42 in seeds else seeds[0]
    for profile, settings in (("baseline", config), ("tuned", tuned)):
        histories[profile] = {}
        for scenario, problem in problems.items():
            histories[profile][scenario] = {}
            for algo, (_, _, key) in SOLVER_REGISTRY.items():
                for seed in seeds:
                    row, history = trial(problem, algo, settings[key], seed, references[scenario])
                    rows.append({"profile": profile, "scenario": scenario, **row})
                    if seed == representative_seed:
                        histories[profile][scenario][algo] = history
            print(f"Benchmarked {profile}: {scenario}", flush=True)
    write_csv(out / "runs.csv", rows)
    summary = summarize(rows, ["profile", "scenario", "algo"])
    write_csv(out / "summary.csv", summary)
    write_json(out / "histories.json", histories)
    write_json(out / "config.baseline.json", config)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_dirty_at_run": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "python": sys.version, "platform": platform.platform(),
        "libraries": {name: importlib.metadata.version(name) for name in ("matplotlib", "numpy", "networkx")},
        "evaluation_seeds": seeds, "tuning_seeds": tuning_seeds,
        "representative_seed": representative_seed,
        "tuning_runs": len(tuning_rows), "evaluation_runs": len(rows),
        "selection_rule": "Lowest tuning mean gap rounded to 9 decimals; then fewest mean objective calls; then variant name.",
        "first_best_step": "First zero-based history index at which the exact final incumbent appears; not proof of convergence.",
        "steps": "History entries minus one; HC accepted moves plus restart starts, SA proposals, beam rounds, GA generations.",
        "evaluation_trace": "Best evaluated cost, including all evaluated candidates; distinct from accepted/current costs.",
        "timing": "Instrumented serial solver runs after warmup, including counting/history overhead; constructor, validation, reference enumeration, and plotting excluded.",
        "tuning_scope": "One parameter at a time on five scenarios; held-out seeds, but the same maps and scenarios.",
    }
    sources = [ROOT / "main.py", ROOT / "experiments" / "config.baseline.json"]
    for folder, pattern in (("algorithms", "*.py"), ("core", "*.py"),
                             ("experiments", "*.py"), ("data/maps", "*.json"),
                             ("data/scenarios", "*.json")):
        sources.extend((ROOT / folder).glob(pattern))
    metadata["source_sha256"] = {
        source.relative_to(ROOT).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sorted(sources)
    }
    write_json(out / "metadata.json", metadata)
    from experiments.plot_results import render_all
    render_all(out, problems, rows, summary, histories, tuning_summary, references)
    return selection


def parse_seeds(value):
    try:
        seeds = [int(seed) for seed in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from error
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be non-empty and unique.")
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "config.baseline.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--tuning-seeds", type=parse_seeds, default=TUNING_SEEDS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = run(config, args.seeds, args.tuning_seeds, args.output)
    print("Selected variants:", {algo: row["variant"] for algo, row in selection.items()})


if __name__ == "__main__":
    main()
