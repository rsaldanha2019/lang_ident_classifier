import argparse
import os
import sys
import optuna
from rich.console import Console
from rich.table import Table
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
import optuna.visualization as vis

def main():
    parser = argparse.ArgumentParser(description="Pareto² Study Analysis - Theory Standard")
    parser.add_argument("--storage", required=True, help="Path to the sqlite file (e.g., study.db)")
    parser.add_argument("--study-name", required=True, help="Name of the Optuna study")
    args = parser.parse_args()

    console = Console()
    
    # 1. Load Study
    storage_path = args.storage if args.storage.startswith("sqlite:///") else f"sqlite:///{args.storage}"
    try:
        study = optuna.load_study(study_name=args.study_name, storage=storage_path)
    except Exception as e:
        console.print(f"[bold red]Error loading study:[/bold red] {e}")
        sys.exit(1)

    # 2. Results Directory
    output_dir = f"analysis_{args.study_name}"
    os.makedirs(output_dir, exist_ok=True)

    # --- STATISTICAL ANALYSIS (CLI Output) ---
    
    # Best Trial Summary
    best_trial = study.best_trial
    console.print(f"\n[bold green]🏆 BEST TRIAL:[/bold green] #{best_trial.number}")
    console.print(f"   [bold]Objective Value:[/bold] {best_trial.value}")
    
    param_table = Table(title="Winning Parameter Configuration", header_style="bold cyan")
    param_table.add_column("Parameter")
    param_table.add_column("Value")
    for k, v in best_trial.params.items():
        param_table.add_row(k, str(v))
    console.print(param_table)

    # fANOVA Importance (Handles Nested Params)
    console.print("\n[bold yellow]🔍 Sensitivity Analysis (fANOVA):[/bold yellow]")
    try:
        importance = get_param_importances(study, evaluator=FanovaImportanceEvaluator())
        imp_table = Table(show_header=True)
        imp_table.add_column("Parameter", style="dim")
        imp_table.add_column("Importance Score", justify="right")
        for p, s in importance.items():
            imp_table.add_row(p, f"{s:.4f}")
        console.print(imp_table)
    except Exception as e:
        console.print(f"[dim red]Importance calculation failed: {e}[/dim red]")

    # Robustness Check (Top 10% vs Best)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        top_n = max(1, len(completed_trials) // 10)
        top_trials = sorted(completed_trials, key=lambda x: x.value)[:top_n]
        avg_top = sum(t.value for t in top_trials) / top_n
        console.print(f"\n[bold blue]📊 Robustness:[/bold blue] Top 10% Mean: {avg_top:.4f} (Gap to best: {abs(best_trial.value - avg_top):.4f})")

    # --- VISUALIZATION EXPORT (Standard Plots) ---
    
    console.print(f"\n[bold]Exporting Theoretical Plots to {output_dir}/...[/bold]")
    
    try:
        # 1. Optimization History (Convergence Theory)
        vis.plot_optimization_history(study).write_html(os.path.join(output_dir, "1_convergence_history.html"))
        
        # 2. Param Importances (Sensitivity Theory)
        vis.plot_param_importances(study, evaluator=FanovaImportanceEvaluator()).write_html(os.path.join(output_dir, "2_param_importance.html"))
        
        # 3. Slice Plot (Local vs Global Optima Theory)
        vis.plot_slice(study).write_html(os.path.join(output_dir, "3_parameter_slices.html"))
        
        # 4. Parallel Coordinate (High-Dimensional Flow/Nested Logic)
        vis.plot_parallel_coordinate(study).write_html(os.path.join(output_dir, "4_nested_paths.html"))
        
        console.print("[green]✅ Success! Open the HTML files in any browser to view interactive charts.[/green]")
    except Exception as e:
        console.print(f"[bold red]Plotting Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
