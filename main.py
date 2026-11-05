"""
main.py
Main entry point to execute, evaluate, and benchmark local search solvers.
"""

import argparse
import json
import random
import sys
import time
from typing import Dict, List, Tuple

from core.graph import RoadNetwork
from core.congestion import GaussianPeakCongestion
from core.problem import CommuteProblem
from core.state import State

from algorithms.hill_climbing import HillClimbingSolver
from algorithms.simulated_annealing import SimulatedAnnealingSolver
from algorithms.beam_search import LocalBeamSearchSolver
from algorithms.genetic_algorithm import GeneticAlgorithmSolver

from visualization.plot_routes import plot_commute_route
from visualization.plot_metrics import plot_convergence


SOLVER_REGISTRY = {
    "hc": ("Hill Climbing", HillClimbingSolver, "hill_climbing"),
    "sa": ("Simulated Annealing", SimulatedAnnealingSolver, "simulated_annealing"),
    "beam": ("Local Beam Search", LocalBeamSearchSolver, "beam_search"),
    "ga": ("Genetic Algorithm", GeneticAlgorithmSolver, "genetic_algorithm")
}


def load_configuration(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_problem(map_path: str, scenario_path: str, problem_config: dict) -> CommuteProblem:
    network = RoadNetwork.load_json(map_path)
    congestion_model = GaussianPeakCongestion.load_json(scenario_path)

    return CommuteProblem(
        network=network,
        congestion_model=congestion_model,
        origin=problem_config.get("origin", "Home"),
        destination=problem_config.get("destination", "Uni"),
        t_min=float(problem_config.get("t_min", 420.0)),
        t_max=float(problem_config.get("t_max", 600.0)),
        t_step=float(problem_config.get("t_step", 5.0))
    )


def execute_solver(name: str, solver_cls, config_params: dict, problem: CommuteProblem) -> Tuple[State, float, float, List[float]]:
    solver = solver_cls(problem, **config_params)
    start_wall = time.perf_counter()
    best_state, best_cost = solver.solve()
    runtime = time.perf_counter() - start_wall
    return best_state, best_cost, runtime, solver.history


def main():
    parser = argparse.ArgumentParser(description="Commute Optimization Local Search Runner")
    parser.add_argument(
        "--algo",
        choices=["hc", "sa", "beam", "ga", "all"],
        default="all",
        help="Algorithm to execute ('all' benchmarks all implemented solvers)"
    )
    parser.add_argument("--map", default="data/maps/toy_graph.json", help="Path to road network JSON")
    parser.add_argument("--scenario", default="data/scenarios/rush_hour.json", help="Path to congestion scenario JSON")
    parser.add_argument("--config", default="config.json", help="Path to hyperparameters config JSON")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--plot-route", action="store_true", help="Display road network and final chosen route")
    parser.add_argument("--plot-metrics", action="store_true", help="Display convergence trajectory curve")
    parser.add_argument("--save-figs", action="store_true", help="Save visualization plots to disk as PNG")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the interactive traffic network viewer with a dynamic time slider"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    config = load_configuration(args.config)
    problem = build_problem(args.map, args.scenario, config.get("problem", {}))

    targets = list(SOLVER_REGISTRY.keys()) if args.algo == "all" else [args.algo]

    histories: Dict[str, List[float]] = {}
    best_results: Dict[str, Tuple[State, float]] = {}

    print("\n" + "=" * 70)
    print(f" COMMUTE LOCAL SEARCH OPTIMIZATION BENCHMARK")
    print(f" Network Origin: {problem.origin} | Destination: {problem.destination}")
    print(f" Time Domain: {problem.t_min}m to {problem.t_max}m (Step: {problem.t_step}m)")
    print("=" * 70)

    if args.interactive:
        from visualization.interactive_traffic import launch_interactive_traffic_viewer
        print("\n>> Launching interactive traffic viewer. Use the slider to explore dynamic congestion.")
        launch_interactive_traffic_viewer(
            network=problem.network,
            congestion_model=problem.congestion_model,
            t_min=problem.t_min,
            t_max=problem.t_max,
            t_step=problem.t_step
        )
        return

    for key in targets:
        # Each solver gets the same independent seed in individual and all runs.
        if args.seed is not None:
            random.seed(args.seed)
        display_name, solver_cls, cfg_key = SOLVER_REGISTRY[key]
        solver_params = config.get(cfg_key, {})

        print(f"\n>> Running: {display_name} ...")
        try:
            best_state, best_cost, runtime, history = execute_solver(
                display_name, solver_cls, solver_params, problem
            )
            histories[display_name] = history
            best_results[display_name] = (best_state, best_cost)

            print(f"   [Done in {runtime:.3f}s]")
            print(f"   Best Departure   : {best_state.format_time()} ({best_state.departure_time:.1f}m)")
            print(f"   Total Travel Time: {best_cost:.2f} minutes")
            print(f"   Best Route       : {' -> '.join(best_state.path)}")

            if args.plot_route:
                out_path = f"route_{key}.png" if args.save_figs else None
                plot_commute_route(problem, best_state, title=f"{display_name} Best Route",
                                   save_path=out_path, show=not args.save_figs)

        except NotImplementedError:
            print(f"   [Skipped] {display_name} is not implemented yet in algorithms/.")

    if args.plot_metrics and histories:
        out_metrics = "convergence_comparison.png" if args.save_figs else None
        plot_convergence(histories, save_path=out_metrics, show=not args.save_figs)

    print("\n" + "=" * 70)
    print(" BENCHMARK SUMMARY")
    print("=" * 70)
    for name, (state, cost) in best_results.items():
        print(f" * {name:<22} : {cost:6.2f} min | Depart: {state.format_time()} | Path: {' -> '.join(state.path)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
