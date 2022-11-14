import time
from pettingzoo.mpe import simple_v2, simple_spread_v2

from code.utils.models import Q_duelling
from code.multiagent.iql_pt import IQL

"""
https://antonai.blog/multi-agent-reinforcement-learning-openais-maddpg/
https://github.com/xuehy/pytorch-maddpg/blob/master/MADDPG.py
https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch <- uses PettingZoo
https://pettingzoo.farama.org/environments/third_party_envs/
"""

def main():
    env = simple_v2.parallel_env()

    iql_agent = IQL(
        environment=env,
        network=Q_duelling,
        learning_starts=20000)

    iql_agent.train(train_steps=100000, report_freq=20)

    ### Render trained agents ###
    env = simple_v2.parallel_env(render_mode="human")
    while True:
        s_t = env.reset()
        while(env.agents):
            a_t = iql_agent.act(s_t)
            s_tp1, r_t, done, trun, _ = env.step(a_t)
            s_t = s_tp1
            env.render()
            time.sleep(0.1)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()
    