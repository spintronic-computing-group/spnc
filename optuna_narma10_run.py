
from optuna_narma10_study import create_study
from optuna_narma10_objective import objective



if __name__ == "__main__":

    hyperparameter_ranges = {
    'Nvirt': (50, 100),
    'gamma': (0.11, 0.12),    
    'h': (0.35, 0.45),
    'm0': (0.003, 0.0035),
    'theta': (0.28, 0.35),
    'num_instances': (3, 7),        
    'deltabeta_range': (-5.0, 5.0), 
    'weight_range': (0.0, 1.0)      
    }



    bias = True



    Ntrain = 2000
    Ntest = 1000

    temp_params = {
        'beta_prime': 20,    # 初始 beta_prime
        'beta_ref': 20,     # 常量 beta_cons
        'step': 1,       # beta_prime 步长
        'beta_left': 18.9,  # 左侧 beta_prime 范围
        'beta_right': 21.1  # 右侧 beta_prime 范围
    }

    # Create a list to collect all the trials
    
    study = create_study()

    study.optimize(
        lambda trial:objective(trial, Ntrain, Ntest, hyperparameter_ranges,  temp_params,True),
        n_trials=100
    )

