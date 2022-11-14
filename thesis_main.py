from collections import deque
import time
from pettingzoo.mpe import simple_v2, simple_tag_v2, simple_spread_v2
from pettingzoo.sisl import multiwalker_v9

# from maddpg_pt_discrete import MADDPG
from code.multiagent.maddpg_pt import MADDPG

from code.multiagent.ma_models import local_Actor, central_Critic

"""
https://antonai.blog/multi-agent-reinforcement-learning-openais-maddpg/
https://github.com/xuehy/pytorch-maddpg/blob/master/MADDPG.py
https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch <- uses PettingZoo
https://pettingzoo.farama.org/environments/third_party_envs/
"""

# try find a co-op discrete env to check performance!
# also try add an argument for making MADDPG discrete or continuous

env = multiwalker_v9.parallel_env(shared_reward=False)
print(env.possible_agents)

maddpg_agent = MADDPG(
    environment=env,
    actor_base=local_Actor,
    critic_base=central_Critic,
    )

n_episodes = 4000

i = 0
r_sum = [0] * len(env.possible_agents)
r_avg = deque(maxlen=50)
for eps in range(n_episodes):

    s_t = env.reset()
    while(env.agents):
        i += 1
        if i == maddpg_agent.learning_starts:
            print('### Beginning training ###')

        if i > maddpg_agent.learning_starts:
            a_t = maddpg_agent.act(s_t)
        else:
            a_t = {agent: env.action_space(agent).sample() for agent in env.possible_agents}            

        s_tp1, r_t, done, trun, _ = env.step(a_t)

        r_sum = [x + y for x, y in zip(r_sum, r_t.values())]

        maddpg_agent.multiagent_store(s_t, a_t, r_t, s_tp1, done)
        s_t = s_tp1

        if len(maddpg_agent) > maddpg_agent.batch_size and i % maddpg_agent.update_every == 0 and i > maddpg_agent.learning_starts:
            maddpg_agent.update()
            maddpg_agent.soft_target_update()

    r_avg.append(r_sum)
    r_sum = [0] * len(env.possible_agents)

    if eps % 25 == 0 and eps > 0:
        agent_performance_list = [round(sum(x) / len(x), 3) for x in zip(*r_avg)]
        agent_performance_dict = {}
        for j, agent in enumerate(env.possible_agents):
            agent_performance_dict[agent] = agent_performance_list[j]

        print(f"EPISODE {eps} DONE, AVERAGE REWARDS: {agent_performance_dict}")

### Render trained agents ###
r_sum = [0] * len(env.possible_agents)
while True:
    s_t = env.reset()
    while(env.agents):
        a_t = maddpg_agent.act(s_t)
        s_tp1, r_t, done, trun, _ = env.step(a_t)
        env.render()
        time.sleep(0.1)
        r_sum = [x + y for x, y in zip(r_sum, r_t.values())]
        s_t = s_tp1

    print(f"REWARD: {r_sum}")
    r_sum = [0] * len(env.possible_agents)
