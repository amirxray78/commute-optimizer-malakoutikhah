"""Headless figures from measured experiment outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import networkx as nx
import numpy as np

from core.state import State
from main import SOLVER_REGISTRY


COLORS = {"hc": "#cf6800", "sa": "#9345b6", "beam": "#2575b7", "ga": "#21864e"}
SHORT = {"hc": "HC", "sa": "SA", "beam": "Beam", "ga": "GA"}


def panels(title):
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    axes[-1, -1].axis("off")
    return fig, axes.flat


def save(fig, path):
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_convergence(out, histories, references, profile, by_evaluations=True):
    scale = "objective evaluations" if by_evaluations else "solver-specific steps"
    fig, axes = panels(f"{profile.title()} convergence | representative seed | {scale}")
    for ax, (scenario, algorithms) in zip(axes, histories[profile].items()):
        for algo, history in algorithms.items():
            if by_evaluations:
                points = history["evaluation_trace"]
                x, y = zip(*points)
            else:
                y = history["best"]
                x = range(len(y))
            ax.step(x, y, where="post", label=SHORT[algo], color=COLORS[algo],
                    linewidth=1.7, linestyle={"hc": "-", "sa": "--", "beam": "-.", "ga": ":"}[algo])
        ax.axhline(references[scenario]["cost"], color="#333333", linewidth=1,
                   linestyle="--", label="Exact reference")
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel(scale)
        ax.set_ylabel("Travel duration (min)")
        ax.set_xscale("log" if by_evaluations else "symlog", **({} if by_evaluations else {"linthresh": 5}))
        ax.grid(alpha=0.25)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.axes[-1].legend(handles, labels, loc="center", frameon=False)
    filename = ("convergence_comparison.png" if profile == "tuned" else "baseline_convergence.png")
    if not by_evaluations:
        filename = "convergence_steps.png"
    save(fig, out / filename)


def plot_quality(out, summary):
    fig, axes = panels("Quality on held-out seeds | baseline vs tuned")
    scenarios = list(dict.fromkeys(row["scenario"] for row in summary))
    for ax, scenario in zip(axes, scenarios):
        for offset, profile, color in [(-0.18, "baseline", "#9099a4"), (0.18, "tuned", "#286a9e")]:
            values = [next(row["mean_gap_pct"] for row in summary
                           if row["profile"] == profile and row["scenario"] == scenario
                           and row["algo"] == algo) for algo in SHORT]
            bars = ax.barh(np.arange(4) + offset, values, height=0.32, label=profile, color=color)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_yticks(range(4), list(SHORT.values()))
        ax.invert_yaxis()
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel("Mean gap to exact reference (%)")
        ax.set_xlim(0, max(1, ax.get_xlim()[1] * 1.2))
        ax.grid(axis="x", alpha=0.2)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.axes[-1].legend(handles, labels, loc="center", frameon=False)
    save(fig, out / "quality_comparison.png")


def plot_sensitivity(out, summary):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    fig.suptitle("One-parameter sensitivity | tuning seeds only | all five scenarios")
    for ax, algo in zip(axes.flat, SHORT):
        rows = [row for row in summary if row["algo"] == algo]
        labels = [row["variant"] for row in rows]
        bars = ax.barh(labels, [row["mean_gap_pct"] for row in rows], color=COLORS[algo])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(SOLVER_REGISTRY[algo][0])
        ax.set_xlabel("Mean optimality gap (%)")
        ax.set_xlim(0, max(1, ax.get_xlim()[1] * 1.2))
        ax.grid(axis="x", alpha=0.2)
    save(fig, out / "parameter_sensitivity.png")


def plot_diagnostics(out, histories):
    fig, axes = panels("Distinct routes in each population | tuned representative seed")
    for ax, (scenario, algorithms) in zip(axes, histories["tuned"].items()):
        for algo in ("beam", "ga"):
            values = algorithms[algo]["path_diversity"]
            ax.step(range(len(values)), values, where="post", label=SHORT[algo], color=COLORS[algo])
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel("Solver-specific round / generation")
        ax.set_ylabel("Distinct routes")
        ax.set_xscale("symlog", linthresh=5)
        ax.grid(alpha=0.25)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.axes[-1].legend(handles, labels, loc="center", frameon=False)
    save(fig, out / "route_diversity.png")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    scenario = "dual_peak_rush_hour"
    history = histories["baseline"][scenario]["sa"]
    axes[0].plot(history["current"], color="#ab79c0", alpha=0.8, label="Current accepted state")
    axes[0].plot(history["best"], color="#303030", linewidth=2, label="Best-so-far")
    axes[0].set_ylabel("Travel duration (min)")
    axes[0].legend()
    axes[0].set_title("SA: uphill excursions and cooling | baseline | dual peak rush hour")
    axes[1].plot(history["temperature"], color=COLORS["sa"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Temperature for next proposal")
    axes[1].set_xlabel("Proposal index (0 = initial state)")
    for ax in axes:
        ax.grid(alpha=0.25)
    save(fig, out / "annealing_trajectory.png")


def plot_route(problem, history, title, path):
    state = State(tuple(history["path"]), history["departure"])
    graph = nx.DiGraph()
    graph.add_nodes_from(problem.network.all_nodes())
    graph.add_edges_from((u, v) for u, v, _ in problem.network.all_edges())
    positions = {node: problem.network.get_coordinates(node) for node in graph}
    traffic = [problem.congestion_model.get_multiplier(u, v, state.departure_time)
               for u, v in graph.edges]
    norm = colors.Normalize(vmin=0, vmax=max(1, max(traffic, default=0)))
    cmap = plt.get_cmap("YlOrRd")
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color=[cmap(norm(x)) for x in traffic],
                           width=2, arrowsize=15, node_size=650, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edges(graph, positions, ax=ax, edgelist=list(zip(state.path, state.path[1:])),
                           edge_color="#205fa5", width=4, arrowsize=20, node_size=650,
                           connectionstyle="arc3,rad=0.08")
    node_colors = ["#75bd8c" if node == problem.origin else "#ee9d93" if node == problem.destination
                   else "#c7def4" if node in state.path else "#e0e3e7" for node in graph]
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=node_colors,
                           edgecolors="#555555", node_size=650)
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=9)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8,
                 label="Traffic factor at departure (snapshot)")
    ax.set_title(f"{title}\nDeparture {state.format_time()} | travel {history['cost']:.2f} min | blue = chosen route")
    ax.margins(0.12)
    ax.set_axis_off()
    save(fig, path)


def render_all(out, problems, rows, summary, histories, tuning_summary, references):
    plot_convergence(out, histories, references, "baseline")
    plot_convergence(out, histories, references, "tuned")
    plot_convergence(out, histories, references, "tuned", by_evaluations=False)
    plot_quality(out, summary)
    plot_sensitivity(out, tuning_summary)
    plot_diagnostics(out, histories)
    routes = out / "routes"
    routes.mkdir(exist_ok=True)
    for scenario, algorithms in histories["tuned"].items():
        for algo, history in algorithms.items():
            plot_route(problems[scenario], history, f"{scenario.replace('_', ' ')} | {SHORT[algo]} | tuned",
                       routes / f"{scenario}_{algo}.png")
