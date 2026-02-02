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

    # 2. DYNAMIC STABILITY TARGET
    volatility = (std_acc / mean_acc) if mean_acc > 0 else 1.0
    dynamic_multiplier = 10 if volatility > 0.1 else 5
    dynamic_target = n_params * dynamic_multiplier
    trials_needed = max(0, dynamic_target - n_completed)
    
    # 3. DOUBLE PARETO LOGIC (0.25 Params / 0.25 Trials)
    importance = get_param_importances(study)
    
    # Selection A: Top 25% of Parameters
    n_top_params = max(2, int(n_params * 0.25)) 
    top_params_list = list(importance.keys())[:n_top_params]
    
    # Selection B: Top 25% of Trials (Consensus Group - minimum of 3)
    n_elite_trials = max(3, int(n_completed * 0.25)) 
    elite_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:n_elite_trials]
    
    convergence_count = 0
    if len(elite_trials) >= 3:
        for p in top_params_list:
            # Agreement check across the top 25% elite trial group
            values_in_elite = [str(t.params.get(p)) for t in elite_trials]
            if len(set(values_in_elite)) == 1:
                convergence_count += 1
    
    # 4. Header Panel
    convergence_ratio = convergence_count / n_top_params
    status_color = "green" if convergence_ratio >= 0.66 else "yellow"
    
    header = Panel(
        f"[bold cyan]Study:[/bold cyan] {STUDY_NAME}\n"
        f"[bold green]Best Value:[/bold green] {best_acc:.4f} | [bold white]Trials:[/bold white] {n_completed}\n"
        f"Convergence Signal: [{status_color}]{'HIGH' if convergence_ratio >= 0.66 else 'LOW'}[/{status_color}] "
        f"({convergence_count}/{n_top_params} Elite Params stable across top {n_elite_trials} trials)",
        title="Pareto² Analytics (25/25 Rule)", border_style="magenta"
    )
    console.print(header)

    # 5. Roadmap & Statistics
    engine_panel = Panel(
        f"Expected Stability: [bold cyan]at trial {dynamic_target}[/bold cyan]\n"
        f"Elite Group Size: [bold yellow]{n_elite_trials}[/bold yellow] trials\n"
        f"Remaining: [bold white]{trials_needed}[/bold white] trials",
        title="Roadmap", border_style="blue"
    )
    stats_panel = Panel(
        f"Best: {best_acc:.4f}\nMean: {mean_acc:.4f}\nStd Dev: {std_acc:.4f}",
        title="Statistics", border_style="magenta"
    )
    console.print(Columns([engine_panel, stats_panel]))

    # 6. Importance Table
    table = Table(title="\nHyperparameter Impact (0.25/0.25 Scaling)", header_style="bold yellow")
    table.add_column("Rank", justify="center")
    table.add_column("Hyperparameter", style="cyan")
    table.add_column("Impact %", justify="right")
    table.add_column("Current Best", style="green")
    table.add_column("Support Status", justify="center")

    for i, (param, score) in enumerate(importance.items(), 1):
        pct = score * 100
        is_elite_param = param in top_params_list
        prefix = "[bold magenta]ELITE[/bold magenta]" if is_elite_param else " "
        
        # Elite Group Consensus Check (Top 25% of Trials)
        vals = [str(t.params.get(param)) for t in elite_trials]
        is_stable = len(set(vals)) == 1
        support = "[bold green]STABLE[/bold green]" if is_stable else "[dim]VARYING[/dim]"
        
        table.add_row(f"{i}", param, f"{pct:.2f}%", str(study.best_params.get(param)), f"{prefix} {support}")

    console.print(table)

    # 7. Strategic Verdict / Reasoning
    reasoning = (
        f"• [bold cyan]Target Adjustment:[/bold cyan] Expected stability at trial [bold]{dynamic_target}[/bold].\n"
        f"• [bold cyan]Why?[/bold cyan] Volatility is [bold]{volatility:.2f}[/bold]. The target scales to filter parallel noise.\n"
        f"• [bold cyan]Asymmetric Logic:[/bold cyan] Monitoring top 25% of params across the elite 25% trial cluster.\n"
        f"• [bold cyan]Convergence:[/bold cyan] You have [bold]{convergence_count}/{n_top_params}[/bold] elite parameters locked in."
    )
    console.print(Panel(reasoning, title="Dynamic Reasoning", border_style="bright_white"))

if __name__ == "__main__":
    main()

