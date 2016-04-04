import numpy as np
import math
import sys
from random import Random
from sets import Set

from q_learning_agent import Agent
from agents import SingleAgentID
from actions import ActionMap
from actions import Action
from actions import NumericActionComponent
from actions import ComponentName
from observations import FeatureName
from binning import discretize


class MappingAgent(Agent):
    feature_names = {
        'x': FeatureName('x'),
        'y': FeatureName('y'),
        'north': FeatureName('north'),
        'east': FeatureName('east'),
        'south': FeatureName('south'),
        'west': FeatureName('west'),
        }

    direction_bases = {
        'north': (0, 1),
        'east': (1, 0),
        'south': (0, -1),
        'west': (-1, 0),
        }

    component_names = {
        'x': ComponentName('x'),
        'y': ComponentName('y'),
        }

    BINS_PER_DIMENSION = (24, 24)

    STARTING_LOG_ODDS_VALUE = 0.0

    def __init__(self, environment_spec):
        action_descriptors = environment_spec.action_descriptors
        self.x_component_descriptor = action_descriptors.descriptors[self.component_names['x']]
        self.y_component_descriptor = action_descriptors.descriptors[self.component_names['y']]

        self.dimensions = (
            self.x_component_descriptor.boundaries()[1].action_value,
            self.y_component_descriptor.boundaries()[1].action_value,
            )

        self.agent_id = SingleAgentID()
        x_bins = self.BINS_PER_DIMENSION[0]
        y_bins = self.BINS_PER_DIMENSION[1]
        self.log_map = np.zeros((x_bins, y_bins)) + self.STARTING_LOG_ODDS_VALUE
        self.observed_map = np.zeros((x_bins, y_bins))
        self.rand = Random()

    def update(self, agent_update):
        observation_map = agent_update.observation_map
        observation = observation_map.observations[self.agent_id]
        self.position = (
            observation.get_value(self.feature_names['x']).feature_value,
            observation.get_value(self.feature_names['y']).feature_value,
            )

        observed_bases = {}
        for direction, basis in zip(self.direction_bases.keys(), self.direction_bases.values()):
            distance = observation.get_value(self.feature_names[direction]).feature_value
            observed_bases[basis] = distance

        self.log_map, self.observed_map = self.update_maps(
            self.position,
            self.log_map,
            self.observed_map,
            observed_bases,
            )

        action_map = ActionMap()
        action = Action()
        next_x, next_y = self.next_movement(self.position, self.log_map)
        x_component = NumericActionComponent(next_x)
        action.add_component(self.component_names['x'], x_component)
        y_component = NumericActionComponent(next_y)
        action.add_component(self.component_names['y'], y_component)
        action_map.add_action(self.agent_id, action)
        return action_map

    def next_movement(self, position, log_map, attempts=0):
        candidate = (
            position[0] + self.rand.uniform(-0.2, 0.2),
            position[1] + self.rand.uniform(-0.2, 0.2),
            )
        disc_candidate = self.discretize_point(candidate)
        log_odds = log_map[disc_candidate[0], disc_candidate[1]]
        proba = self.proba_from_log_odds(log_odds)
        if self.rand.uniform(0, 1) < proba or attempts > 100:
            return candidate
        else:
            return self.next_movement(position, log_map, attempts + 1)

    def update_maps(self, position, log_map, observed_map, observed_bases):
        position_in_grid = self.discretize_point(position)
        for basis, distance in zip(observed_bases.keys(), observed_bases.values()):
            theta_from_horizontal = self.angle_between(basis, (1, 0))
            endpoint = (
                position[0] + distance * math.cos(theta_from_horizontal),
                position[1] + distance * math.sin(theta_from_horizontal),
                )
            endpoint_in_grid = self.discretize_point(endpoint)
            bins_crossed = self.bins_crossed(position_in_grid, endpoint_in_grid)

            for bin_location in bins_crossed:
                log_map, observed_map = self.update_bin(
                    bin_location,
                    log_map,
                    observed_map,
                    False,
                    )
            log_map, observed_map = self.update_bin(
                endpoint_in_grid,
                log_map,
                observed_map,
                True,
                )
        return log_map, observed_map

    def update_bin(self, bin_location, log_map, observed_map, obstacle_encountered):
        x = bin_location[0]
        y = bin_location[1]

        observed_map[x, y] = 1
        log_map[x, y] = log_map[x, y] + self.inverse_sensor(
            bin_location,
            obstacle_encountered,
            ) - self.STARTING_LOG_ODDS_VALUE
        return log_map, observed_map
        
    def inverse_sensor(self, bin_location, obstacle_encountered):
        if obstacle_encountered:
            proba = 0.8
        else:
            proba = 0.2

        return math.log(proba / (1 - proba))

    def bins_crossed(self, position_in_grid, endpoint_in_grid):
        """
        IF I WANT DIAGONAL SENSES
        I HAVE TO SIGNIFICANTLY IMPROVE THIS            
        Does not include endpoint in return set"""
        bins_crossed = Set()
        if position_in_grid[0] == endpoint_in_grid[0]:
            # movement is in y direction
            for y_coord in self.get_range(
                position_in_grid[1],
                endpoint_in_grid[1],
                ):
                    bins_crossed.add((position_in_grid[0], y_coord))
        elif position_in_grid[1] == endpoint_in_grid[1]:
            # movement is in x direction
            for x_coord in self.get_range(
                position_in_grid[0],
                endpoint_in_grid[0],
                ):
                    bins_crossed.add((x_coord, position_in_grid[1]))

        else:
            raise ValueError("Diagonal movement")

        return bins_crossed

    def get_range(self, pos_coord, end_coord):
        if pos_coord == end_coord:
            return []
        elif pos_coord < end_coord:
            return xrange(pos_coord, end_coord)
        else:
            return xrange(end_coord + 1, pos_coord + 1)

    def proba_from_log_odds(self, log_odds):
        return 1 - (1/(1 + math.exp(log_odds)))

    def discretize_point(self, point):
        return (
            self.discretize_in_direction(point[0], 0),
            self.discretize_in_direction(point[1], 1),
            )
            
    def discretize_in_direction(self, val, direction):
        return discretize(
            val,
            0,
            self.dimensions[direction],
            self.BINS_PER_DIMENSION[direction],
            )

    def angle_between(self, vec1, vec2):
        return math.acos(np.dot(vec1, vec2) / (math.sqrt(vec1[0] ** 2 + vec1[1] ** 2) * math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)))

    def proba_map(self):
        return np.array([map(self.proba_from_log_odds, row) for row in self.log_map])
