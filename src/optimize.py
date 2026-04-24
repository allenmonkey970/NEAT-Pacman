import optuna
import configparser
from Pacman import run_neat

BASE_CONFIG = "config/neat_config.txt"
WORK_CONFIG = "config/neat_config_trial.txt"


def update_config(base_file, work_file, params):
    config = configparser.ConfigParser()
    config.read(base_file)
    config['NEAT']['pop_size'] = str(params['pop_size'])
    config['DefaultSpeciesSet']['compatibility_threshold'] = str(params['compatibility_threshold'])
    config['DefaultGenome']['conn_add_prob'] = str(params['conn_add_prob'])
    config['DefaultGenome']['conn_delete_prob'] = str(params['conn_delete_prob'])
    config['DefaultGenome']['weight_mutate_rate'] = str(params['weight_mutate_rate'])
    config['DefaultGenome']['node_add_prob'] = str(params['node_add_prob'])
    config['DefaultGenome']['node_delete_prob'] = str(params['node_delete_prob'])
    with open(work_file, 'w') as f:
        config.write(f)


def objective(trial):
    params = {
        "pop_size": trial.suggest_int("pop_size", 50, 200, step=25),
        "compatibility_threshold": trial.suggest_float("compatibility_threshold", 2.0, 5.0, step=0.5),
        "conn_add_prob": trial.suggest_float("conn_add_prob", 0.3, 0.8, step=0.05),
        "conn_delete_prob": trial.suggest_float("conn_delete_prob", 0.1, 0.8, step=0.05),
        "weight_mutate_rate": trial.suggest_float("weight_mutate_rate", 0.5, 1.0, step=0.05),
        "node_add_prob": trial.suggest_float("node_add_prob", 0.1, 0.8, step=0.05),
        "node_delete_prob": trial.suggest_float("node_delete_prob", 0.05, 0.5, step=0.05),
    }
    print(f"Trial {trial.number} params: {params}")
    update_config(BASE_CONFIG, WORK_CONFIG, params)
    score = run_neat(config_path=WORK_CONFIG)
    print(f"Trial {trial.number} score: {score}")
    return score


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open("outputs/best_optuna_params.txt", "w") as f:
        f.write(f"Best trial value: {trial.value}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
    print("Saved to outputs/best_optuna_params.txt")


if __name__ == "__main__":
    main()
