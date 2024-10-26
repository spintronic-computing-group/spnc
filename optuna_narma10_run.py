
from optuna_narma10_study import create_study, run_study



if __name__ == "__main__":


    params = {'theta': 0.3,'gamma' : .113,'delay_feedback' : 0,'Nvirt' : 400}

    bias = True

    hyperparameter_ranges = {
        'Nvirt': (300, 400),
        'gamma': (0.038, 0.05),    
        'h': (0.43, 0.48),
        'm0': (0.002, 0.003),
        'theta': (0.23, 0.25),
        'num_instances': (2, 3),        
        'deltabeta_range': (-5.0, 5.0), 
        'weight_range': (0.0, 1.0)      
    }

    Ntrain = 1
    Ntest = 2000
    temp_params = {
        'beta_prime': 35,    # 初始 beta_prime
        'beta_ref': 35,     # 常量 beta_cons
        'step': 0.875,       # beta_prime 步长
        'beta_left': 33.25,  # 左侧 beta_prime 范围
        'beta_right': 36.75  # 右侧 beta_prime 范围
    }

    # Create a list to collect all the trials
    

    study = create_study()

    all_pred_and_Stest, best_beta, best_nrmse = run_study(study, hyperparameter_ranges,  temp_params, n_trials=50)