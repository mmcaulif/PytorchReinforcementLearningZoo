from collections import deque
import matplotlib.pyplot as plt
import numpy as np

class MultiagentRewardLogger():
    def __init__(self, agents, episode_avg, report_freq):
        # Logging variables
        self.agents = agents
        self.running_length = episode_avg
        self.report_freq = report_freq

        # Metrics
        self.r_trajectory = {agent: 0 for agent in self.agents}
        self.r_avg = {agent: deque(maxlen=self.running_length) for agent in self.agents}

        # Plotting variables
        self.r_plot = {agent: [] for agent in self.agents}
        self.episodes = 0
        self.steps = 0

    def log_step(self, rewards, agents):
        self.steps += 1
        for agent in agents:
            self.r_trajectory[agent] += rewards[agent]              
            
        if self.report_freq is not None and self.steps % self.report_freq == 0:
            stats = self.stats()
            avg_stats = sum(stats.values())/len(stats)
            print(f"TIMESTEPS {self.steps} DONE, EPISODES {self.episodes} DONE, AVERAGE EPISODIC REWARDS: {avg_stats}")
        pass
        
    def end_of_eps(self):
        self.episodes += 1
        for agent in self.agents:
            self.r_avg[agent].append(self.r_trajectory[agent])
            self.r_plot[agent].append(round(sum(self.r_avg[agent])/len(self.r_avg[agent]), 2))
            self.r_trajectory[agent] = 0

        pass

    def stats(self):
        stats_dict = {}
        for agent in self.agents:
            stats_dict[agent] = round(sum(self.r_avg[agent])/len(self.r_avg[agent]), 2)
        return stats_dict

    def plot_training(self):
        plt.plot(range(self.episodes), self.r_plot[self.agents[0]])
        plt.xlabel("Episodes")
        plt.ylabel(f"Average episodic reward over {self.running_length} episodes")
        if max(self.r_plot[self.agents[0]]) > 200:
            plt.ylim([-200, 0])
        # plt.yscale('symlog')  # alternative option
        plt.show()
        pass

class RewardLogger():
    def __init__(self, episode_avg=40, report_freq=20):
        # Logging variables
        self.running_length = episode_avg
        self.report_freq = report_freq

        # Metrics
        self.r_trajectory = 0
        self.r_avg = deque(maxlen=self.running_length)
        self.best = -np.Inf

        # Plotting variables
        self.r_plot = []
        self.steps = 0
        self.episodes = 0

    def log_step(self, rewards):
        self.r_trajectory += rewards
        self.steps += 1       
            
        if self.report_freq is not None and self.steps % self.report_freq == 0:
            print(f"TIMESTEPS {self.steps} DONE, EPISODES {self.episodes} DONE, AVERAGE EPISODIC REWARDS: {self.stats()}")
        pass
        
    def end_of_eps(self):
        self.episodes += 1
        self.r_avg.append(self.r_trajectory)
        self.r_plot.append(round(sum(self.r_avg)/len(self.r_avg), 2))
        self.r_trajectory = 0
        
        if self.stats() > self.best:
            self.best = self.stats()

        pass

    def stats(self):
        stats = round(sum(self.r_avg)/len(self.r_avg), 2)
        return stats

    def plot_training(self):
        plt.plot(range(self.episodes), self.r_plot)
        plt.xlabel("Episodes")
        plt.ylabel(f"Average episodic reward over {self.running_length} episodes")
        plt.show()
        pass