import numpy as np
import optuna
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spnc_ml import spnc_narma10_heterogenous

def objective(trial: optuna.Trial, hyperparameter_ranges: dict,temp_params: dict,bias = bias, params = params) -> tuple[float, float]:

    # pick up the hyperparameters from the dictionary

    Nvirt_min, Nvirt_max = hyperparameter_ranges['Nvirt']
    gamma_min, gamma_max = hyperparameter_ranges['gamma']
    h_min, h_max = hyperparameter_ranges['h']
    m0_min, m0_max = hyperparameter_ranges['m0']
    theta_min, theta_max = hyperparameter_ranges['theta']
    num_instances_min, num_instances_max = hyperparameter_ranges['num_instances']
    deltabeta_min, deltabeta_max = hyperparameter_ranges['deltabeta_range']
    weight_min, weight_max = hyperparameter_ranges['weight_range']

    # set the hyperparameter space

    Nvirt = trial.suggest_int('Nvirt', Nvirt_min, Nvirt_max)
    gamma = trial.suggest_float('gamma', gamma_min, gamma_max)
    h = trial.suggest_float('h', h_min, h_max)
    m0 = trial.suggest_float('m0', m0_min, m0_max)
    theta = trial.suggest_float('theta', theta_min, theta_max)

    # set the number and proportion of heterogenous reservoir

    num_instances = trial.suggest_int('num_instances', num_instances_min, num_instances_max)
    deltabeta_list = [trial.suggest_float(f'deltabeta{i+1}', deltabeta_min, deltabeta_max) for i in range(num_instances)]
    weights = [trial.suggest_float(f'weight_{i}', weight_min, weight_max) for i in range(num_instances)]

    # nomalize weights

    total_weight = sum(weights)
    if total_weight > 0:
        weights = [round(w / total_weight, 3) for w in weights]
    total_weight = sum(weights)
    if not np.isclose(total_weight, 1.0, atol=1e-3):
        diff = 1.0 - total_weight
        min_index = weights.index(min(weights))
        weights[min_index] = round(weights[min_index] + diff, 3)
    assert np.isclose(sum(weights), 1.0, atol=1e-3)

    # pick up parameters from the dictionary

    beta_prime = temp_params['beta_prime']
    beta_ref = temp_params['beta_ref']
    step = temp_params['step']
    beta_left = temp_params['beta_left']
    beta_right = temp_params['beta_right']

    # run in the range of temperature

    beta_prime, nrmse = spnc_narma10_heterogenous(1, 2000, Nvirt, gamma, beta_prime, beta_ref, deltabeta_list, h, theta, m0, step, beta_left, beta_right, *weights, bias=bias, params=params, seed_NARMA=1234)

    model_performance = np.mean(nrmse)
    thermal_stability = np.std(nrmse)

    return float(model_performance), float(thermal_stability)

