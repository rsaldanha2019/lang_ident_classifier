import optuna
import numpy as np
import argparse
import sys
from optuna.importance import FanovaImportanceEvaluator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

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
    parser = argparse.ArgumentParser(description="Pareto² Study Analysis")
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name", required=True)
    args = parser.parse_args()

    console = Console()
    try:
        study = optuna.load_study(study_name=args.study_name, storage="sqlite:///"+args.storage)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}"); sys.exit(1)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed: return

    # 1. CORE MATH
    values = [t.value for t in completed if t.value is not None]
    best_acc = study.best_value
    mean_acc, std_acc = np.mean(values), np.std(values)
    volatility = (std_acc / mean_acc) if mean_acc > 0 else 0

    # 2. IMPORTANCE & HIERARCHY
    # fANOVA is used for synchronized rank, but we manually fetch all param names 
    # to ensure zero-importance parameters aren't hidden.
    importance_scores = optuna.importance.get_param_importances(study, evaluator=FanovaImportanceEvaluator())
    all_param_names = sorted(list(set().union(*(t.params.keys() for t in completed))))
    hierarchy_map = get_hierarchy_map(study, completed)

    # 3. DOUBLE PARETO LOGIC (25/25)
    n_params = len(all_param_names)
    dynamic_target = n_params * (10 if volatility > 0.1 else 5)
    n_elite_trials = max(3, int(len(completed) * 0.25))
    elite_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:n_elite_trials]
    
    # Sort by importance, then by name for parameters with 0.0 importance
    sorted_params = sorted(all_param_names, key=lambda x: (importance_scores.get(x, 0), x), reverse=True)
    n_top_params = max(2, int(n_params * 0.25))
    top_params_list = sorted_params[:n_top_params]

    # Check stability only for parameters present in elite trials
    convergence_count = 0
    for p in top_params_list:
        vals = [str(t.params.get(p)) for t in elite_trials if p in t.params]
        if vals and len(set(vals)) == 1:
            convergence_count += 1

    # 4. RENDER UI - Header & Roadmap/Stats Panels
    status_color = "green" if (convergence_count/n_top_params) >= 0.66 else "yellow"
    console.print(Panel(
        f"[bold cyan]Study:[/bold cyan] {args.study_name}\n"
        f"[bold green]Best Value:[/bold green] {best_acc:.4f} | [bold white]Trials:[/bold white] {len(completed)}\n"
        f"Convergence Signal: [{status_color}]{'HIGH' if (convergence_count/n_top_params) >= 0.66 else 'LOW'}[/{status_color}] "
        f"({convergence_count}/{n_top_params} Elite Params stable across top {n_elite_trials} trials)",
        title="Pareto² Analytics (25/25 Rule)", border_style="magenta"
    ))

    console.print(Columns([
        Panel(f"Expected Stability: [bold cyan]at trial {int(dynamic_target)}[/bold cyan]\n"
              f"Elite Group Size: [bold yellow]{n_elite_trials}[/bold yellow] trials\n"
              f"Remaining: [bold white]{max(0, int(dynamic_target - len(completed)))}[/bold white] trials", title="Roadmap", border_style="blue"),
        Panel(f"Best: {best_acc:.4f}\nMean: {mean_acc:.4f}\nStd Dev: {std_acc:.4f}", title="Statistics", border_style="magenta")
    ]))

    # 5. IMPORTANCE TABLE
    table = Table(title="\nHyperparameter Impact (fANOVA Synchronized)", header_style="bold yellow")
    table.add_column("Rank", justify="center")
    table.add_column("Hyperparameter", style="cyan")
    table.add_column("Hierarchy", style="white")
    table.add_column("Impact %", justify="right")
    table.add_column("Current Best", style="green")
    table.add_column("Support Status", justify="center")

    for i, param in enumerate(sorted_params, 1):
        score = importance_scores.get(param, 0.0)
        vals = [str(t.params.get(param)) for t in elite_trials if param in t.params]
        
        # Determine Support Status text
        if not vals:
            support = "[red]NOT IN ELITE[/red]"
        else:
            is_stable = len(set(vals)) == 1
            support = "[bold green]STABLE[/bold green]" if is_stable else "[dim]VARYING[/dim]"
        
        elite_tag = "[bold magenta]ELITE[/bold magenta]" if param in top_params_list else " "
        
        table.add_row(
            str(i), param, hierarchy_map.get(param, "Global"),
            f"{score*100:.2f}%", str(study.best_params.get(param, "N/A")), 
            f"{elite_tag} {support}"
        )
    console.print(table)

    # 6. DYNAMIC REASONING
    reasoning = (
        f"• [bold cyan]Evaluation Engine:[/bold cyan] fANOVA (Accounting for nested/conditional variance).\n"
        f"• [bold cyan]Target Adjustment:[/bold cyan] Expected stability at trial [bold]{int(dynamic_target)}[/bold].\n"
        f"• [bold cyan]Why?[/bold cyan] Volatility is [bold]{volatility:.2f}[/bold]. The target scales to filter parallel noise.\n"
        f"• [bold cyan]Asymmetric Logic:[/bold cyan] Monitoring top 25% of params across the elite 25% trial cluster.\n"
        f"• [bold cyan]Hierarchy Detection:[/bold cyan] Parameters not in all trials are flagged as [white]Sub-parameters[/white].\n"
        f"• [bold cyan]Convergence:[/bold cyan] You have [bold]{convergence_count}/{n_top_params}[/bold] elite parameters locked in."
    )
    console.print(Panel(reasoning, title="Dynamic Reasoning", border_style="bright_white"))

if __name__ == "__main__":
    main()
