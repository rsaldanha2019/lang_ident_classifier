import optuna
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

STORAGE = "sqlite:///optim_studies/lang_ident_classifier_optim_ai4bharat/study.db"
STUDY_NAME = "lang_ident_classifier_optim_ai4bharat"

def main():
    console = Console()
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}"); return

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        console.print("[yellow]No completed trials found.[/yellow]"); return

    importance = optuna.importance.get_param_importances(study)
    
    table = Table(title="Dynamic Driver Analysis", header_style="bold yellow")
    table.add_column("Rank", justify="center")
    table.add_column("Hyperparameter", style="cyan")
    table.add_column("Importance %", justify="right")
    table.add_column("Best Group Mean", justify="right", style="green")
    table.add_column("Worst Group Mean", justify="right", style="red")
    table.add_column("Impact Swing", justify="left")

    # Store results for the verdict
    results_summary = []

    for i, (param, score) in enumerate(importance.items(), 1):
        param_values = {}
        for t in completed:
            val = t.params.get(param)
            if val is not None:
                if val not in param_values: param_values[val] = []
                param_values[val].append(t.value)

        if not param_values: continue

        cat_means = {k: np.mean(v) for k, v in param_values.items()}
        best_mean = max(cat_means.values())
        worst_mean = min(cat_means.values())
        delta = best_mean - worst_mean

        # Dynamic bar scaling based on delta
        bar = "█" * int(delta * 40)
        table.add_row(str(i), param, f"{score*100:.1f}%", f"{best_mean:.4f}", f"{worst_mean:.4f}", f"{bar} (+{delta:.4f})")
        
        results_summary.append({"name": param, "score": score, "delta": delta})

    console.print(table)

    # Dynamic Verdict Logic
    top_3 = results_summary[:3]
    verdict_text = ""
    for item in top_3:
        impact_type = "Critical" if item['score'] > 0.2 else "Moderate"
        verdict_text += f"• [bold cyan]{item['name']}[/bold cyan]: {impact_type} driver ([bold]{item['score']*100:.1f}%[/bold] variance importance).\n"

    console.print(Panel(verdict_text, title="Automated Driver Verdict", border_style="bright_white"))

if __name__ == "__main__":
    main()
