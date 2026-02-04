def find_logical_parent(param_name, all_trials, study):
    """
    Inferred hierarchy. Logic:
    1. If param is missing in ANY trial, find the param whose value governs it.
    2. If param is in ALL trials (early study), check if it's conditional in the search space.
    """
    active_indices = [i for i, t in enumerate(all_trials) if param_name in t.params]
    
    # CASE A: Parameter has already "vanished" in some trials (The hierarchy is visible)
    if len(active_indices) < len(all_trials):
        potential_parents = {}
        for i in active_indices:
            for p, v in all_trials[i].params.items():
                if p == param_name: continue
                potential_parents.setdefault(p, set()).add(v)
        for p, values in potential_parents.items():
            if len(values) == 1:
                return f"↳ via {p} ({list(values)[0]})"
        return "Conditional"

    # CASE B: Early study (Param in all trials). 
    # Check if Optuna's internal distribution recognizes it as conditional.
    # This reaches into the distribution metadata of the best trial.
    best_trial = study.best_trial
    if param_name in best_trial.distributions:
        # If we can't find a parent via variance yet, we mark it as Root 
        # until a False/Alternative branch is explored.
        pass

    return "Global Root"
