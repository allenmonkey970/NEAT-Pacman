import optuna
import configparser
import shutil
import os
from Pacman import run_neat

# Path to original NEAT config file
BASE_CONFIG = "neat_config.txt"
WORK_CONFIG = "neat_config_bo.txt"

def update_config(base_file, work_file, params):
    config = configparser.ConfigParser()
    config.read(base_file)

    # Update only the params you want to optimize
    config['NEAT']['pop_size'] = str(params['pop_size'])
    config['DefaultSpeciesSet']['compatibility_threshold'] = str(params['compatibility_threshold'])
    config['DefaultGenome']['conn_add_prob'] = str(params['conn_add_prob'])
    config['DefaultGenome']['conn_delete_prob'] = str(params['conn_delete_prob'])
    config['DefaultGenome']['weight_mutate_rate'] = str(params['weight_mutate_rate'])
    config['DefaultGenome']['node_add_prob'] = str(params['node_add_prob'])
    config['DefaultGenome']['node_delete_prob'] = str(params['node_delete_prob'])
    config['NEAT']['fitness_threshold'] = str(params['fitness_threshold'])

    with open(work_file, 'w') as f:
        config.write(f)

def objective(trial):
    params = {
        "pop_size": trial.suggest_int("pop_size", 50, 200, step=25),
        "compatibility_threshold": trial.suggest_float("compatibility_threshold", 1.0, 3.0, step=0.1),
        "conn_add_prob": trial.suggest_float("conn_add_prob", 0.3, 0.8, step=0.05),
        "conn_delete_prob": trial.suggest_float("conn_delete_prob", 0.1, 0.8, step=0.05),
        "weight_mutate_rate": trial.suggest_float("weight_mutate_rate", 0.5, 1.0, step=0.05),
        "node_add_prob": trial.suggest_float("node_add_prob", 0.1, 0.8, step=0.05),
        "node_delete_prob": trial.suggest_float("node_delete_prob", 0.05, 0.5, step=0.05),
        "fitness_threshold": trial.suggest_float("fitness_threshold", 1.0, 3.0, step=0.1),
    }
    print(f"Trying params: {params}")

    # Update config file for trial
    update_config(BASE_CONFIG, WORK_CONFIG, params)

    # Run NEAT with the new config
    score = run_neat(config_path=WORK_CONFIG)
    print(f"Score: {score}")

    # Optuna tries to maximize this value
    return score

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best params to a file
    with open("best_bayes_params.txt", "w") as f:
        f.write(f"Best trial value: {trial.value}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()