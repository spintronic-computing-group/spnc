
from optuna_narma10_study import create_study
from optuna_narma10_spnc_heterogenous import objective



if __name__ == "__main__":


    params = {'theta': 0.3,'gamma' : .113,'delay_feedback' : 0,'Nvirt' : 400}

    bias = True

    hyperparameter_ranges = {
        'Nvirt': (400, 450),
        'gamma': (0.111, 0.112),    
        'h': (0.39, 0.41),
        'm0': (0.003, 0.0031),
        'theta': (0.3, 0.301),
        'num_instances': (1, 2),        
        'deltabeta_range': (-5.0, 5.0), 
        'weight_range': (0.0, 1.0)      
    }

    Ntrain = 2000
    Ntest = 1000

    temp_params = {
        'beta_prime': 35,    # 初始 beta_prime
        'beta_ref': 35,     # 常量 beta_cons
        'step': 0.875,       # beta_prime 步长
        'beta_left': 33.25,  # 左侧 beta_prime 范围
        'beta_right': 36.75  # 右侧 beta_prime 范围
    }

    # Create a list to collect all the trials
    
    study = create_study()

    study.optimize(
        lambda trial:objective(trial, hyperparameter_ranges,  temp_params,True, params),
        n_trials=2
    )

