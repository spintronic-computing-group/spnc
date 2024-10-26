import optuna
import threading
from optuna.samplers import TPESampler
from optuna_narma10_spnc_heterogenous import objective  
from optuna_dashboard import run_dashboard

import matplotlib.pyplot as plt

def create_study():

    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()

    """创建或删除旧的 study 并新建 study"""
    suffix = 0
    storage = "sqlite:///db.sqlite3" 
    study_name = "MOO_test"
    new_study_name = study_name
    
    while True:
        try:
            # try to load the study
            optuna.load_study(study_name=new_study_name, storage=storage)
            # if the study exists, add a suffix to the study name
            suffix += 1
            new_study_name = f"{study_name}_{suffix}"
        except KeyError:
            # if the study does not exist, break the loop
            print(f"Study '{new_study_name}' 不存在，创建新的 study。")
            break
            
    # set up the study
    study = optuna.create_study(
        sampler=TPESampler(),
        directions=["minimize", "minimize"],  
        storage=storage,
        study_name=new_study_name,
    )
    
    return study

def run_study(study, hyperparameter_ranges, temp_params, n_trials=50):
    all_temp_and_nrmse = []
    all_pred_and_dtest = {}
    study.optimize(
    lambda trial:objective(trial, hyperparameter_ranges,  temp_params),
    n_trials=n_trials
    )
    pareto_front_trail = [trial for trial in study.best_trials]
    pareto_front_data = [data for data in all_temp_and_nrmse if data['trial_number'] in pareto_front_trail]
    best_beta = [data['valid_beta_primes'] for data in pareto_front_data]
    best_nrmse = [data['valid_nrmse_values'] for data in pareto_front_data]

    return all_pred_and_dtest, best_beta, best_nrmse