import gym
import optuna
from tqdm import trange

from code.utils.models import Q_quantregression
from code.distributional.qrdqn_pt import QR_DQN

def objective(trial):
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    n = trial.suggest_int("Dirac interval", 16, 32)
    ls = 1000   # trial.suggest_categorical("Learning_starts", [0, 250, 1000, 5000, 10000])
    tf = trial.suggest_int("Train frequency", 1, 16)
    bs = 32 # trial.suggest_categorical("Batch size", [16, 32, 64, 128])
    tu = trial.suggest_categorical("Target update frequency", [200, 300, 500, 1000])
    lr = trial.suggest_float("Learning rate", 1e-5, 1e-1, log=True)

    qrdqn_agent = QR_DQN(
        env,
        Q_quantregression(env.observation_space.shape[0], env.action_space.n, hidden_dims=64),
        learning_starts=ls,
        train_freq=tf,
        batch_size=bs,
        target_update=tu,
        lr=lr,
        N=n)

    n_runs = 3

    end_result = 0
    for _ in trange(n_runs):
        end_result+= qrdqn_agent.train(train_steps=50000, report_freq=None)

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

    """
    Best trial:
        Value:  416.6333333333334
        Params: 
            Dirac interval: 30
            Train frequency: 2
            Target update frequency: 300
            Learning rate: 0.0009446247996866446
    """

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()