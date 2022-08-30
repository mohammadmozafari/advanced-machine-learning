import copy
import random
import pickle
import os
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

class Trainer(object):
    """Runs games for given agents. Optionally will visualise and save the results"""
    def __init__(self, config, agents):
        self.config = config
        self.agents = agents
        self.agent_to_agent_group = self.create_agent_to_agent_group_dictionary()
        self.results = None

    def create_agent_to_agent_group_dictionary(self):
        """Creates a dictionary that maps an agent to their wider agent group"""
        agent_to_agent_group_dictionary = {
            "DQN": "DQN_Agents",
            "DDQN": "DQN_Agents",
            "DQN with Fixed Q Targets": "DQN_Agents",
            "SAC": "Actor_Critic_Agents",
            "DIAYN": "DIAYN",
        }
        return agent_to_agent_group_dictionary

    def run_games_for_agents(self):
        """Run a set of games for each agent. Optionally visualising and/or saving the results"""
        self.results = self.create_object_to_store_results()
        for agent_number, agent_class in enumerate(self.agents):
            agent_name = agent_class.agent_name
            agent = self.run_games_for_agent(agent_number + 1, agent_class) # stores results in self.results
            self.agents[agent_number] = agent
        plt.show()
        return self.results

    def create_object_to_store_results(self):
        """Creates a dictionary that we will store the results in if it doesn't exist, otherwise it loads it up"""
        if self.config.overwrite_existing_results_file or not self.config.file_to_save_data_results or not os.path.isfile(self.config.file_to_save_data_results):
            results = {}
        else: results = self.load_obj(self.config.file_to_save_data_results)
        return results

    def run_games_for_agent(self, agent_number, agent_class):
        """Runs a set of games for a given agent, saving the results in self.results"""
        agent_results = []
        agent_name = agent_class.agent_name
        agent_group = self.agent_to_agent_group[agent_name]
        agent_round = 1
        agents = []
        for run in range(self.config.runs_per_agent):
            agent_config = copy.deepcopy(self.config)
            agent_config.environment = self.config.environment
            if self.config.randomise_random_seed: agent_config.seed = random.randint(0, 2**32 - 2)
            print("AGENT NAME: {}".format(agent_name))
            agent_config.hyperparameters = agent_config.hyperparameters[agent_group]
            print("\033[1m" + "{}.{}: {}".format(agent_number, agent_round, agent_name) + "\033[0m", flush=True)
            agent = agent_class(agent_config)
            self.environment_name = agent.environment_title
            print(agent.hyperparameters)
            print("RANDOM SEED " , agent_config.seed)
            game_scores, rolling_scores, time_taken = agent.run_n_episodes()
            print("Time taken: {}".format(time_taken), flush=True)
            print("-----------------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------------")
            print(" ")
            agent_results.append([game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores), time_taken])
            agent_round += 1
            agents.append(agent)
        self.results[agent_name] = agent_results
        return agents

    def get_mean_and_standard_deviation_difference_results(self, results):
        """From a list of lists of agent results it extracts the mean results and the mean results plus or minus
         some multiple of the standard deviation"""
        def get_results_at_a_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return results_at_a_time_step
        def get_standard_deviation_at_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return np.std(results_at_a_time_step)
        mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(len(results[0]))]
        mean_minus_x_std = [mean_val - self.config.standard_deviation_results * get_standard_deviation_at_time_step(results, timestep) for
                            timestep, mean_val in enumerate(mean_results)]
        mean_plus_x_std = [mean_val + self.config.standard_deviation_results * get_standard_deviation_at_time_step(results, timestep) for
                           timestep, mean_val in enumerate(mean_results)]
        return mean_minus_x_std, mean_results, mean_plus_x_std

    def hide_spines(self, ax, spines_to_hide):
        """Hides splines on a matplotlib image"""
        for spine in spines_to_hide:
            ax.spines[spine].set_visible(False)
