
from optuna_narma10_study import create_study
from optuna_narma10_objective import objective
import optuna
from optuna_narma10_callback import callback
from contextmanager import suppress_stdout


'''
Try to add the function of callback to this version.

The purpose of callback is to control the termination of the study under certain conditions.

'''



if __name__ == "__main__":

    verbose = False

    hyperparameter_ranges = {
    'Nvirt': (50,600),
    'gamma': (0.05, 0.2),    
    'h': (0.2, 0.6),
    'm0': (0.001, 0.005),
    'theta': (0.1, 0.5),
    'num_instances': (2, 8),        
    'deltabeta_range': (-5.0, 5.0), 
    'weight_range': (0.0, 1.0)      
    }

    bias = True

    Ntrain = 2000
    Ntest = 1000

    temp_params = {
        'beta_prime': 20,    # initial beta_prime
        'beta_ref': 20,     # reference beta_cons
        'step': 0.55,       # beta_prime step
        'beta_left': 18.9,  # left beta_prime range
        'beta_right': 21.1  # right beta_prime range
    }

    with suppress_stdout(suppress=verbose):
        study = create_study()

        # study_name = 'MOO_test_30'
        # storage_name = "sqlite:///db.sqlite3"
        # study = optuna.load_study(study_name=study_name, storage=storage_name)
        target = (0.6, 0.05)
        callback = callback(target)

        study.optimize(
        lambda trial:objective(trial, Ntrain, Ntest, hyperparameter_ranges,  temp_params,True),
        n_trials=None, callbacks=[callback]
        )

    # Print
    print('Best trial:')
    print('  Value: ', study.best_trial.value)
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')
