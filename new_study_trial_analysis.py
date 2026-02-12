#!/usr/bin/env python3

# FINAL — STABLE
# Slice plot rendered via Matplotlib for clean UI
# All other plots remain unchanged
# This WILL RUN

import argparse
import os
import sys
import math
import optuna
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
from optuna.trial import TrialState
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from rich.console import Console
from rich.table import Table
import optuna.visualization as vis


# ------------------------------------------------------------
# VISUAL PATCHES
# ------------------------------------------------------------

def darken_markers(fig):
    fig.update_traces(
        marker=dict(line=dict(width=1.5, color="black")),
        selector=dict(type="scatter")
    )
    fig.update_traces(
        line=dict(width=2),
        selector=dict(type="box")
    )
    return fig


# ------------------------------------------------------------
# SLICE PLOT — MATPLOTLIB (CLEAN & READABLE)
# FUNCTION NAME KEPT
# ------------------------------------------------------------

def add_pseudo_legend_and_counts(study, outpath):
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    params = sorted({k for t in trials for k in t.params})
    cols = 4
    rows = math.ceil(len(params) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, param in zip(axes, params):
        values = [t.params[param] for t in trials if param in t.params]
        scores = [t.value for t in trials if param in t.params]

        uniq = sorted(set(values))
        index = {v: i for i, v in enumerate(uniq)}
        x = [index[v] for v in values]

        ax.scatter(
            x,
            scores,
            edgecolors="black",
            facecolors="tab:blue",
            alpha=0.6
        )

        ax.set_title(param, pad=20)
        # Raise subplot title
        ax.set_ylabel("Objective")
        ax.set_xticks(range(len(uniq)))
        ax.set_xticklabels(range(len(uniq)))

        legend_text = "\n".join(f"{i}: {v}" for i, v in enumerate(uniq))
        ax.text(
            0.5, -0.35,
            legend_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="black")
        )

        counts = Counter(values)
        for v, c in counts.items():
            ax.text(
                index[v],
                1.04, # Raise n= value higher
                f"n={c}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(fc="white", ec="none", alpha=0.8),
                clip_on=False
            )

    for ax in axes[len(params):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name", required=True)
    args = parser.parse_args()

    console = Console()
    storage = args.storage if args.storage.startswith("sqlite:///") else f"sqlite:///{args.storage}"

    try:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
    except Exception as e:
        console.print(f"[bold red]Error loading study:[/bold red] {e}")
        sys.exit(1)

    outdir = f"analysis_{args.study_name}"
    os.makedirs(outdir, exist_ok=True)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == TrialState.FAIL]
    best = study.best_trial

    console.print("\n[bold cyan]STUDY OVERVIEW[/bold cyan]")
    overview = Table(show_header=False, box=None)
    overview.add_row("Direction:", study.direction.name)
    overview.add_row("Sampler:", study.sampler.__class__.__name__)
    overview.add_row("Total Trials:", str(len(study.trials)))
    overview.add_row("Completed:", str(len(completed)))
    overview.add_row("Pruned:", str(len(pruned)))
    overview.add_row("Failed:", str(len(failed)))
    overview.add_row("Best Trial:", f"#{best.number}")
    overview.add_row("Best Value:", f"{best.value:.6f}")
    console.print(overview)

    param_table = Table(title="Best Configuration Parameters")
    param_table.add_column("Parameter")
    param_table.add_column("Value")
    for k, v in best.params.items():
        param_table.add_row(k, str(v))
    console.print(param_table)

    importance = get_param_importances(study, evaluator=FanovaImportanceEvaluator())
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    imp_table = Table(title="Parameter Importance (fANOVA)")
    imp_table.add_column("Parameter")
    imp_table.add_column("Importance", justify="right")
    for p, s in importance.items():
        imp_table.add_row(p, f"{s*100:.2f}%")
    console.print(imp_table)

    console.print(f"\n[bold]Exporting plots to {outdir}/[/bold]")

    fig = vis.plot_optimization_history(study)
    darken_markers(fig).write_html(os.path.join(outdir, "convergence.html"))

    add_pseudo_legend_and_counts(
        study,
        os.path.join(outdir, "parameter_slices.png")
    )

    fig = vis.plot_parallel_coordinate(study, params=list(best.params.keys()))
    darken_markers(fig).write_html(os.path.join(outdir, "multidim_paths.html"))

    fig = go.Figure(
        go.Bar(
            x=list(importance.values()),
            y=list(importance.keys()),
            orientation="h",
            marker=dict(line=dict(width=1.5, color="black"))
        )
    )

    fig.update_layout(
        title="Hyperparameter Importances (fANOVA)",
        xaxis_title="Importance",
        yaxis_title="Hyperparameter",
        yaxis=dict(autorange="reversed")
    )

    fig.write_html(os.path.join(outdir, "param_importance.html"))


    if len(importance) >= 2:
        fig = vis.plot_contour(study, params=list(importance.keys())[:2])
        darken_markers(fig).write_html(os.path.join(outdir, "top_param_interactions.html"))

    console.print("[bold green]DONE[/bold green]")


if __name__ == "__main__":
    main()

