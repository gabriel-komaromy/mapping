import math
import sys
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from messages import EpisodeTerminationSignal
from messages import FreezeExploration
from messages import FreezeLearning
from messages import AgentUpdate
from messages import EnvironmentUpdate
from environments import Environment
from dill_io import save_dill

from forget_map_context import forget_map_context
from merge_maps import merge_maps

class Experiment(object):
    def __init__(self, agent_class_name, agent_module_path, mdp_class_name, mdp_module_path):
        agent_class = import_from_strings(agent_class_name, agent_module_path)
        mdp_class = import_from_strings(mdp_class_name, mdp_module_path)
        dummy_environment = Environment(mdp_class((1, 1)))
        self.spec = dummy_environment.spec()
        self.agent_class = agent_class
        self.mdp_class = mdp_class

    def run_episodes(self, num_cycles, explore_per_cycle, exploit_per_cycle, steps_per_episode):
        maps = []
        with open('robot_start_positions.txt', 'r') as positions:
            positions_strings = positions.readlines()
        split_coordinates = [position.split(',') for position in positions_strings]
        position_coordinates = []
        for position in split_coordinates:
            position_coordinates.append([float(coord.strip()) + 1 for coord in position])
        for start_location in position_coordinates:
            # print start_location
            agent = self.agent_class(self.spec)
            episode = Episode(agent, self.mdp_class, FreezeExploration(False), FreezeLearning(False), start_location)
            episode.run(steps_per_episode)
            free_map = forget_map_context(agent.proba_map().T, agent.observed_map.T)
            maps.append(free_map)
            # self.plot_map(agent.proba_map().T, start_location)
        """
        for _ in xrange(num_cycles):
            for _ in xrange(explore_per_cycle):
                episode = Episode(self.agent, self.mdp_class, FreezeExploration(False), FreezeLearning(False))
                episode.run(steps_per_episode)
            for _ in xrange(exploit_per_cycle):
                episode = Episode(self.agent, self.mdp_class, FreezeExploration(True), FreezeLearning(True))
                cumulative_reward = episode.run(steps_per_episode)
                rewards_received.append(cumulative_reward)
        self.save_results(rewards_received, 'saved_results/mc_q_learning.dill')
        return rewards_received
        """

        merge_maps(maps)

    def save_policy(self, agent, file_name):
        policy = agent.policy()
        save_dill(policy, file_name)

    def save_results(self, output_data, file_name):
        save_dill(output_data, file_name)

    def plot_map(self, to_plot, start_location):
        fig, ax = plt.subplots()
        ax.imshow(to_plot, cmap=cm.gray, interpolation='nearest')
        numrows, numcols = to_plot.shape
        def format_coord(x, y):
            col = int(x)
            row = int(y)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = to_plot[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        ax.format_coord = format_coord
        tick_points = np.arange(-0.5, 49.5, 5.0)
        labels = map(lambda tick: str(int(tick + 0.5)), tick_points)
        plt.xticks(tick_points, labels)
        plt.yticks(tick_points, labels)
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        plt.gca().invert_yaxis()
        # plt.title(start_location)
        plt.show()


class Episode(object):
    def __init__(self, agent, mdp_class, freeze_exploration, freeze_learning, robot_position):
        self.agent = agent
        self.environment = Environment(mdp_class(robot_position))
        self.freeze_exploration = freeze_exploration
        self.freeze_learning = freeze_learning

    def run(self, num_steps):
        freeze_exploration = self.freeze_exploration
        freeze_learning = self.freeze_learning
        agent = self.agent
        environment = self.environment
        state = environment.initial_state()
        for _ in xrange(num_steps):
            observation_map = state.observation_map
            reward = state.reward

            termination_signal = state.termination_signal
            new_agent_update = AgentUpdate(
                observation_map,
                reward,
                termination_signal,
                freeze_learning,
                freeze_exploration,
                )
            agent_output = agent.update(new_agent_update)
            if termination_signal:
                break
            new_environment_update = EnvironmentUpdate(
                agent_output,
                EpisodeTerminationSignal(False),
                )
            state = environment.update_environment(new_environment_update)
        return 0


def import_from_strings(class_name, module_path):
    # http://stackoverflow.com/a/547867/3737529
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


if __name__ == '__main__':
    experiment = Experiment('MappingAgent', 'mapping_agent', 'WallWorldMDP', 'wall_world_mdp')
    experiment.run_episodes(10, 5, 5, 1000)
