import optuna

'''
this function is used to control the termination of the study under certain conditions.

'''


def callback(study, trial):
    target_performance = 0.6  # Target for the first objective
    target_stability = 0.05   # Target for the second objective

    # Access the best trial's values for each objective
    best_performance, best_stability = study.best_trials[0].values

    # Stop the study if both objectives meet their respective targets
    if best_performance <= target_performance and best_stability <= target_stability:
        study.stop()

