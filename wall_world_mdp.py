import math

from messages import EnvironmentSpec
from messages import EpisodeTerminationSignal
from environments import MarkovDecisionProcess
from actions import AllActionDescriptors
from actions import ComponentName
from actions import ContinuousComponentDescriptor
from actions import NumericActionComponent
from agents import SingleAgentID
from observations import Reward
from observations import AllFeatureDescriptors
from observations import FeatureName
from observations import ContinuousFeatureDescriptor
from observations import NumericFeatureValue
from observations import ObservationMap
from observations import Observation

from wall_world import World

class WallWorldMDP(MarkovDecisionProcess):
    feature_names = {
        'x': FeatureName('x'),
        'y': FeatureName('y'),
        'north': FeatureName('north'),
        'east': FeatureName('east'),
        'south': FeatureName('south'),
        'west': FeatureName('west'),
        }

    def __init__(self, robot_position):
        self.dimensions = (12, 12)
        self.world = World(self.dimensions, self.feature_names, robot_position)
        self.reward = Reward(None)
        self.termination_signal = EpisodeTerminationSignal(False)
        self.agent = SingleAgentID()

    def spec(self):
        action_descriptors = AllActionDescriptors()
        x_component_name = ComponentName('x')
        min_x_component = NumericActionComponent(0)
        max_x_component = NumericActionComponent(self.dimensions[0])
        x_component_descriptor = ContinuousComponentDescriptor(min_x_component, max_x_component)
        action_descriptors.add_descriptor(x_component_name, x_component_descriptor)

        y_component_name = ComponentName('y')
        min_y_component = NumericActionComponent(0)
        max_y_component = NumericActionComponent(self.dimensions[1])
        y_component_descriptor = ContinuousComponentDescriptor(min_y_component, max_y_component)

        action_descriptors.add_descriptor(y_component_name, y_component_descriptor)

        feature_descriptors = AllFeatureDescriptors()
        min_x_value = NumericFeatureValue(0)
        max_x_value = NumericFeatureValue(self.dimensions[0])
        x_feature_descriptor = ContinuousFeatureDescriptor(min_x_value, max_x_value)
        feature_descriptors.add_descriptor(self.feature_names['x'], x_feature_descriptor)
        min_y_value = NumericFeatureValue(0)
        max_y_value = NumericFeatureValue(self.dimensions[1])
        y_feature_descriptor = ContinuousFeatureDescriptor(min_y_value, max_y_value)
        feature_descriptors.add_descriptor(self.feature_names['y'], y_feature_descriptor)

        farthest_distance = math.sqrt(self.dimensions[0] ** 2 + self.dimensions[1] ** 2)
        for name in ['north', 'east', 'south', 'west']:
            min_value = NumericFeatureValue(0)
            max_value = NumericFeatureValue(farthest_distance)
            direction_descriptor = ContinuousFeatureDescriptor(min_value, max_value)
            feature_descriptors.add_descriptor(self.feature_names[name], direction_descriptor)
        spec = EnvironmentSpec(action_descriptors, feature_descriptors)
        return spec

    def initial_state(self):
        observation_map = ObservationMap()
        observation = self.world.initial_state()
        observation_map.add_observation(self.agent, observation)
        return observation_map, self.reward, self.termination_signal

    def update(self, action_map, term_signal):
        observation_map = ObservationMap()
        observation = self.world.update(action_map, term_signal)
        observation_map.add_observation(self.agent, observation)
        return observation_map, self.reward, self.termination_signal
