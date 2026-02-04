import optuna
import numpy as np
import argparse
import sys
from optuna.importance import FanovaImportanceEvaluator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def get_hierarchy_map(study, completed_trials):
    """
    Detects nesting: If Param B only exists when Param A exists, B is a sub-parameter.
    """
    all_params = sorted(list(set().union(*(t.params.keys() for t in completed_trials))))
    hierarchy = {p: "[dim]Global[/dim]" for p in all_params}
    presence = {p: set(t.number for t in completed_trials if p in t.params) for p in all_params}
    total_completed = len(completed_trials)

    for child in all_params:
        child_mask = presence[child]
        if len(child_mask) == total_completed: continue
        
        for parent in all_params:
            if child == parent: continue
            parent_mask = presence[parent]
            if child_mask.issubset(parent_mask) and len(parent_mask) > len(child_mask):
                hierarchy[child] = f"Sub of [bold]{parent}[/bold]"
                break
    return hierarchy

def main():
    # 1. COMPULSORY COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(description="Dynamic Driver Analysis")
    parser.add_argument("--storage", required=True, help="Database URI (e.g., study.db)")
    parser.add_argument("--study-name", required=True, help="Name of the Optuna study")
    args = parser.parse_args()

    console = Console()
    try:
        # Load Study with fANOVA support
        study = optuna.load_study(study_name=args.study_name, storage="sqlite:///"+args.storage)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        console.print("[yellow]No completed trials found.[/yellow]"); return

    # 2. CALCULATE IMPORTANCE (The Anchor for Synchronization)
    importance = optuna.importance.get_param_importances(
        study, 
        evaluator=FanovaImportanceEvaluator()
    )
    
    # Get hierarchy for the new column
    hierarchy_map = get_hierarchy_map(study, completed)
    
    # 3. BUILD TABLE
    table = Table(title="Dynamic Driver Analysis (Synchronized Rank)", header_style="bold yellow")
    table.add_column("Rank", justify="center")
    table.add_column("Hyperparameter", style="cyan")
    table.add_column("Hierarchy", style="white") 
    table.add_column("Importance %", justify="right")
    table.add_column("Best Group Mean", justify="right", style="green")
    table.add_column("Worst Group Mean", justify="right", style="red")
    table.add_column("Impact Swing", justify="left")

    results_summary = []

    # 4. ANALYZE SWINGS (Iterating through importance order)
    for i, (param, score) in enumerate(importance.items(), 1):
        param_values = {}
        for t in completed:
            val = t.params.get(param)
            if val is not None:
                if val not in param_values: param_values[val] = []
                param_values[val].append(t.value)

        if not param_values: continue

        # Group stats
        cat_means = {k: np.mean(v) for k, v in param_values.items()}
        best_mean = max(cat_means.values())
        worst_mean = min(cat_means.values())
        delta = best_mean - worst_mean

        # Dynamic bar scaling
        bar_len = int(delta * 35) 
        bar = "█" * bar_len
        
        table.add_row(
            str(i), 
            param, 
            hierarchy_map.get(param, "Global"),
            f"{score*100:.1f}%", 
            f"{best_mean:.4f}", 
            f"{worst_mean:.4f}", 
            f"{bar} (+{delta:.4f})"
        )
        
        results_summary.append({"name": param, "score": score, "delta": delta})

    console.print(table)

    # 5. DYNAMIC VERDICT LOGIC
    top_3 = results_summary[:3]
    verdict_text = ""
    for item in top_3:
        if item['score'] > 0.20:
            impact_type = "Critical"
        elif item['score'] > 0.10:
            impact_type = "High-Impact"
        else:
            impact_type = "Moderate"
            
        verdict_text += f"• [bold cyan]{item['name']}[/bold cyan]: {impact_type} driver ([bold]{item['score']*100:.1f}%[/bold] variance importance).\n"

    # SENSITIVITY CHECK: Identify hidden volatility
    if results_summary:
        max_swing_param = max(results_summary, key=lambda x: x['delta'])
        if max_swing_param['name'] not in [t['name'] for t in top_3]:
            verdict_text += f"\n[yellow]Note:[/yellow] [bold cyan]{max_swing_param['name']}[/bold cyan] has the highest [bold]Impact Swing (+{max_swing_param['delta']:.4f})[/bold] despite lower overall importance."

    if not verdict_text:
        verdict_text = "• [yellow]Insufficient variance detected to isolate primary drivers.[/yellow]"

    console.print(Panel(verdict_text, title="Automated Driver Verdict", border_style="bright_white"))

if __name__ == "__main__":
    main()
