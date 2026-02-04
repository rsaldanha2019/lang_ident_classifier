import optuna
import numpy as np
from optuna.importance import FanovaImportanceEvaluator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

STORAGE = "sqlite:///optim_studies/lang_ident_classifier_optim_ai4bharat/study.db"
STUDY_NAME = "lang_ident_classifier_optim_ai4bharat"

def main():
    console = Console()
    try:
        # 1. Load Study with fANOVA support
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}"); return

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        console.print("[yellow]No completed trials found.[/yellow]"); return

    # 2. Calculate Importance using fANOVA (Better for conditional/categorical)
    # This addresses the 'Best Trial' basis by looking at variance drivers
    importance = optuna.importance.get_param_importances(
        study, 
        evaluator=FanovaImportanceEvaluator()
    )
    
    table = Table(title="Dynamic Driver Analysis", header_style="bold yellow")
    table.add_column("Rank", justify="center")
    table.add_column("Hyperparameter", style="cyan")
    table.add_column("Importance %", justify="right")
    table.add_column("Best Group Mean", justify="right", style="green")
    table.add_column("Worst Group Mean", justify="right", style="red")
    table.add_column("Impact Swing", justify="left")

    results_summary = []

    # 3. Analyze Swings
    for i, (param, score) in enumerate(importance.items(), 1):
        param_values = {}
        for t in completed:
            val = t.params.get(param)
            if val is not None:
                if val not in param_values: param_values[val] = []
                param_values[val].append(t.value)

        if not param_values: continue

        # Group stats: Determine which value of the parameter performed best vs worst
        cat_means = {k: np.mean(v) for k, v in param_values.items()}
        best_mean = max(cat_means.values())
        worst_mean = min(cat_means.values())
        delta = best_mean - worst_mean

        # Dynamic bar scaling: normalization against max possible swing (1.0)
        bar_len = int(delta * 35) # Adjusted for table width
        bar = "█" * bar_len
        
        table.add_row(
            str(i), 
            param, 
            f"{score*100:.1f}%", 
            f"{best_mean:.4f}", 
            f"{worst_mean:.4f}", 
            f"{bar} (+{delta:.4f})"
        )
        
        results_summary.append({"name": param, "score": score, "delta": delta})

    console.print(table)

    # 4. Dynamic Verdict Logic
    # Filters based on fANOVA importance to identify the real movers
    top_3 = results_summary[:3]
    verdict_text = ""
    for item in top_3:
        # Driver classification based on variance contribution
        if item['score'] > 0.20:
            impact_type = "Critical"
        elif item['score'] > 0.10:
            impact_type = "High-Impact"
        else:
            impact_type = "Moderate"
            
        verdict_text += f"• [bold cyan]{item['name']}[/bold cyan]: {impact_type} driver ([bold]{item['score']*100:.1f}%[/bold] variance importance).\n"

    # Fill if less than 3 drivers identified
    if not verdict_text:
        verdict_text = "• [yellow]Insufficient variance detected to isolate primary drivers.[/yellow]"

    console.print(Panel(verdict_text, title="Automated Driver Verdict", border_style="bright_white"))

if __name__ == "__main__":
    main()
