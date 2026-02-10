import argparse
import os
import sys
import optuna
from rich.console import Console
from rich.table import Table
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
import optuna.visualization as vis

def main():
    parser = argparse.ArgumentParser(description="Optuna Study Analysis - Deep Convergence Proof")
    parser.add_argument("--storage", required=True, help="Path to sqlite file (e.g., study.db)")
    parser.add_argument("--study-name", required=True, help="Name of the Optuna study")
    args = parser.parse_args()

    console = Console()
    storage_path = args.storage if args.storage.startswith("sqlite:///") else f"sqlite:///{args.storage}"
    
    try:
        study = optuna.load_study(study_name=args.study_name, storage=storage_path)
    except Exception as e:
        console.print(f"[bold red]Error loading study:[/bold red] {e}")
        sys.exit(1)

    output_dir = f"analysis_{args.study_name}"
    os.makedirs(output_dir, exist_ok=True)

    # --- METADATA & SAMPLER ---
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_trial = study.best_trial
    sampler_name = study.sampler.__class__.__name__

    console.print(f"\n[bold cyan]STUDY OVERVIEW[/bold cyan]")
    overview_table = Table(show_header=False, box=None)
    overview_table.add_row("Sampler Used:", f"[bold white]{sampler_name}[/bold white]")
    overview_table.add_row("Total Trials:", f"[bold white]{len(study.trials)}[/bold white]")
    overview_table.add_row("Completed Trials:", f"[bold white]{len(completed_trials)}[/bold white]")
    overview_table.add_row("Best Trial Number:", f"[bold green]#{best_trial.number}[/bold green]")
    overview_table.add_row("Best Objective Value:", f"[bold yellow]{best_trial.value}[/bold yellow]")
    console.print(overview_table)

    # --- BEST CONFIGURATION ---
    param_table = Table(title="Best Configuration Parameters", header_style="bold magenta")
    param_table.add_column("Parameter")
    param_table.add_column("Value")
    for k, v in best_trial.params.items():
        param_table.add_row(k, str(v))
    console.print(param_table)

    # --- THE PROOF: STATISTICAL VALIDATION ---
    console.print("\n[bold cyan]STATISTICAL PROOF & SUPPORT[/bold cyan]")
    
    # 1. Hyperparameter Importance (fANOVA)
    importance = None  # Initialize to ensure it exists for visualization logic
    try:
        importance = get_param_importances(study, evaluator=FanovaImportanceEvaluator())
        imp_table = Table(title="Parameter Sensitivity (fANOVA)", show_header=True)
        imp_table.add_column("Parameter")
        imp_table.add_column("Importance", justify="right")
        for p, s in importance.items():
            imp_table.add_row(p, f"{s*100:.2f}%")
        console.print(imp_table)
    except Exception as e:
        console.print(f"[red]Importance Proof Failed: {e}[/red]")

    # 2. Stability/Robustness (Top 10% Analysis)
    if len(completed_trials) > 1:
        top_n = max(1, len(completed_trials) // 10)
        top_trials = sorted(completed_trials, key=lambda x: x.value)[:top_n]
        avg_top = sum(t.value for t in top_trials) / top_n
        std_top = (sum((t.value - avg_top)**2 for t in top_trials) / top_n)**0.5
        
        robustness_table = Table(title="Robustness Analysis (Top 10% Trials)", box=None)
        robustness_table.add_row("Mean of Top 10%:", f"{avg_top:.6f}")
        robustness_table.add_row("Standard Deviation:", f"{std_top:.6f}")
        robustness_table.add_row("Gap (Best vs Top 10%):", f"{abs(best_trial.value - avg_top):.6f}")
        console.print(robustness_table)

    # --- VISUALIZATION EXPORT ---
    console.print(f"\n[bold]Exporting Analysis Plots to {output_dir}/...[/bold]")
    try:
        # Plot 1: Optimization History
        vis.plot_optimization_history(study).write_html(os.path.join(output_dir, "convergence.html"))
        
        # Plot 2: Slice Plot
        vis.plot_slice(study).write_html(os.path.join(output_dir, "parameter_slices.html"))
        
        # Plot 3: Parallel Coordinate
        vis.plot_parallel_coordinate(study).write_html(os.path.join(output_dir, "multidim_paths.html"))
        
        # Plot 4: Parameter Importance (Integrated here)
        vis.plot_param_importances(study, evaluator=FanovaImportanceEvaluator()).write_html(os.path.join(output_dir, "param_importance.html"))
        
        # Plot 5: Contour
        if len(best_trial.params) >= 2:
            # Uses the importance dict created in the fANOVA section above
            top_params = list(importance.keys())[:2] if importance is not None else None
            vis.plot_contour(study, params=top_params).write_html(os.path.join(output_dir, "top_param_interactions.html"))

        console.print("[green]Analysis Complete. View HTML files for visual proof.[/green]")
    except Exception as e:
        console.print(f"[bold red]Visualization Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
