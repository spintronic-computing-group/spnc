
from optuna_narma10_study import create_study
from optuna_narma10_objective import objective
import optuna



if __name__ == "__main__":

    hyperparameter_ranges = {
    'Nvirt': (50, 400),
    'gamma': (0.05, 0.2),    
    'h': (0.3, 0.5),
    'm0': (0.002, 0.004),
    'theta': (0.23, 0.35),
    'num_instances': (2, 7),        
    'deltabeta_range': (-5.0, 5.0), 
    'weight_range': (0.0, 1.0)      
    }

    bias = True

    Ntrain = 2000
    Ntest = 1000

    temp_params = {
        'beta_prime': 20,    # initial beta_prime
        'beta_ref': 20,     # reference beta_cons
        'step': 1.5,       # beta_prime step
        'beta_left': 18.9,  # left beta_prime range
        'beta_right': 21.1  # right beta_prime range
    }

    study = create_study()

    # study_name = 'MOO_test_30'
    # storage_name = "sqlite:///db.sqlite3"
    # study = optuna.load_study(study_name=study_name, storage=storage_name)

    study.optimize(
    lambda trial:objective(trial, Ntrain, Ntest, hyperparameter_ranges,  temp_params,True),
    n_trials=50
    )


