import gym

from code.distributional.qrddpg_pt import QR_DDPG
from code.utils.models import Critic_quantregression, td3_Actor

def main():
    env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'
    env = gym.make(env_name)
    
    qrddpg_agent = QR_DDPG(
        env,
        td3_Actor(env.observation_space.shape[0], env.action_space.shape[0]),
        Critic_quantregression(env.observation_space.shape[0], env.action_space.shape[0]), 
        lr=1e-3,
        tau=0.01,
        learning_starts=10000,
        gamma=0.98,)

    qrddpg_agent.train(train_steps=50000, report_freq=1000)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()