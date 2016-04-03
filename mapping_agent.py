import numpy as np

from q_learning_agent import Agent
from agents import SingleAgentID
from actions import ActionMap
from actions import Action
from actions import NumericActionComponent
from actions import ComponentName
from observations import FeatureName


class MappingAgent(Agent):
    feature_names = {
        'x': FeatureName('x'),
        'y': FeatureName('y'),
        'north': FeatureName('north'),
        'east': FeatureName('east'),
        'south': FeatureName('south'),
        'west': FeatureName('west'),
        }

    component_names = {
        'x': ComponentName('x'),
        'y': ComponentName('y'),
        }

    BINS_PER_DIMENSION = 30
    def __init__(self, environment_spec):
        action_descriptors = environment_spec.action_descriptors
        self.x_component_descriptor = action_descriptors.descriptors[self.component_names['x']]
        self.y_component_descriptor = action_descriptors.descriptors[self.component_names['y']]

        self.dimensions = (self.x_component_descriptor.boundaries()[1], self.y_component_descriptor.boundaries()[1])

        self.agent_id = SingleAgentID()
        self.proba_map = np.zeros((self.BINS_PER_DIMENSION, self.BINS_PER_DIMENSION))
        self.observed__map = np.zeros((self.BINS_PER_DIMENSION, self.BINS_PER_DIMENSION))

    def update(self, agent_update):
        observation_map = agent_update.observation_map
        observation = observation_map.observations[self.agent_id]
        self.position = (observation.get_value(self.feature_names['x']), observation.get_value(self.feature_names['y']))
        action_map = ActionMap()
        action = Action()
        next_x, next_y = self.next_movement()
        x_component = NumericActionComponent(next_x)
        action.add_component(self.component_names['x'], x_component)
        y_component = NumericActionComponent(next_y)
        action.add_component(self.component_names['y'], y_component)
        action_map.add_action(self.agent_id, action)
        return action_map

    def next_movement(self):
        return (0.5, 7)
