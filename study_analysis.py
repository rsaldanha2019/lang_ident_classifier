import optuna
import numpy as np
from optuna.importance import get_param_importances
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

STORAGE = "sqlite:///optim_studies/lang_ident_classifier_optim_ai4bharat/study.db"
STUDY_NAME = "lang_ident_classifier_optim_ai4bharat"

def main():
    console = Console()
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}"); return

    # 1. Calculations & Metrics
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in completed if t.value is not None]
    n_completed = len(completed)
    n_params = len(study.trials[0].params) if completed else 12
    
    best_acc = study.best_value
    mean_acc = np.mean(values) if values else 0
    std_acc = np.std(values) if values else 0

    # 2. DYNAMIC STABILITY LOGIC
    # Instead of a fixed 10x, we look at the 'Volatility Index'
    volatility = (std_acc / mean_acc) if mean_acc > 0 else 1.0
    
    # If volatility is low (<0.05), we only need 5x params. 
    # If high (>0.2), we need the full 10x or 15x.
    dynamic_multiplier = 10 if volatility > 0.1 else 5
    dynamic_target = n_params * dynamic_multiplier
    trials_needed = max(0, dynamic_target - n_completed)
    
    # Check for early convergence signal
    importance = get_param_importances(study)
    top_params = list(importance.keys())[:3]
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:3]
    
    # If top 3 parameters are STABLE across top 3 trials, we are near convergence
    convergence_score = 0
    if len(top_trials) >= 3:
        for p in top_params:
            if len(set([str(t.params.get(p)) for t in top_trials])) == 1:
                convergence_score += 1
    
    # 3. Header Panel
    status_color = "green" if convergence_score >= 2 else "yellow"
    header = Panel(
        f"[bold cyan]Study:[/bold cyan] {STUDY_NAME}\n"
    status_color = "green" if convergence_score >= 2 else "yellow"
    header = Panel(
        f"[bold cyan]Study:[/bold cyan] {STUDY_NAME}\n"
        f"[bold green]Best Value:[/bold green] {best_acc:.4f} | [bold white]Trials:[/bold white] {n_completed}\n"
        f"Convergence Signal: [{status_color}]{'HIGH' if convergence_score >= 2 else 'LOW'}[/{status_color}] "
        f"(Volatility: {volatility:.2f})",
        title="Dynamic HPO Analytics", border_style="magenta"
    )
    console.print(header)

    # 4. Roadmap (Dynamic Target)
    engine_panel = Panel(
        f"Dynamic Target: [bold cyan]{dynamic_target}[/bold cyan] trials\n"
        f"Needed: [bold yellow]{trials_needed}[/bold yellow]\n"
        f"Basis: {'High Volatility' if dynamic_multiplier == 10 else 'Stable Signal'}",
        title="Roadmap", border_style="blue"
    )
    stats_panel = Panel(
        f"Best: {best_acc:.4f}\nMean: {mean_acc:.4f}\nStd Dev: {std_acc:.4f}",
        title="Statistics", border_style="magenta"
    )
    console.print(Columns([engine_panel, stats_panel]))

    # 5. Importance Table
    table = Table(title="\nHyperparameter Impact", header_style="bold yellow")
    table.add_column("Rank", justify="center")
    table.add_column("Hyperparameter", style="cyan")
    table.add_column("Impact %", justify="right")
    table.add_column("Visualization")
    table.add_column("Current Best", style="green")
    table.add_column("Support", justify="center")

    for i, (param, score) in enumerate(importance.items(), 1):
        pct = score * 100
        bar = "█" * int(pct / 5)
        # Support check
        is_stable = len(set([str(t.params.get(param)) for t in top_trials])) == 1 if len(top_trials) > 1 else False
        support = "[bold green]STABLE[/bold green]" if is_stable else "[dim]SEARCHING[/dim]"
        table.add_row(str(i), param, f"{pct:.2f}%", f"[magenta]{bar}[/magenta]", str(study.best_params.get(param)), support)

    console.print(table)

    # 6. Reasoning Section
    reasoning = (
        f"• [bold cyan]Target Adjustment:[/bold cyan] Your current target is set to [bold]{dynamic_target}[/bold] trials.\n"
        f"• [bold cyan]Why?[/bold cyan] " + 
        ("High volatility detected. We need more samples to ensure the Best Value isn't an outlier." if dynamic_multiplier == 10 else 
         "The signal is stabilizing. We've reduced the target because the top parameters are consistent.") + "\n"
        f"• [bold cyan]Convergence:[/bold cyan] You have [bold]{convergence_score}/3[/bold] top parameters locked in."
    )
    console.print(Panel(reasoning, title="Dynamic Reasoning", border_style="bright_white"))

if __name__ == "__main__":
    main()

