import gym
import optuna
from tqdm import trange

from code.utils.models import Q_quantregression
from code.distributional.qrdqn_pt import QR_DQN

def objective(trial):
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    n = trial.suggest_int("Dirac interval", 16, 32)
    ls = trial.suggest_categorical("Learning_starts", [0, 250, 1000, 5000, 10000])
    tf = trial.suggest_int("Train frequency", 1, 16)
    bs = trial.suggest_categorical("Batch size", [16, 32, 64, 128])
    tu = trial.suggest_categorical("Target update frequency", [200, 300, 500, 1000])
    lr = trial.suggest_float("Learning rate", 1e-5, 1e-1, log=True)

    c51_agent = QR_DQN(
        env,
        Q_quantregression,
        learning_starts=ls,
        train_freq=tf,
        batch_size=bs,
        target_update=tu,
        learning_rate=lr,
        N=n)

    n_runs = 2

    end_result = 0
    for _ in trange(n_runs):
        end_result+= c51_agent.train(train_steps=50000, report_freq=None)

    return end_result/n_runs

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()