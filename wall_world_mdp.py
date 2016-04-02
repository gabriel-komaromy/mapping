from environments import MarkovDecisionProcess
from actions import ComponentName
from actions import ContinuousComponentDescriptor
from actions import AllActionDescriptors
from actions import NumericActionComponent
from wall_world import World

class WallWorldMDP(MarkovDecisionProcess):
    def __init__(self):
        self.dimensions = (10, 10)
        self.world = World(self.dimensions)

    def spec(self):
        action_descriptors = AllActionDescriptors()
        x_component_name = ComponentName('x')
        min_x_component = NumericActionComponent(0)
        max_x_component = self.dimensions[0]
        x_component_descriptor = ContinuousComponentDescriptor(min_x_component, max_x_component)
        action_descriptors.add_descriptor(x_component_name, x_component_descriptor)

        y_component_name = ComponentName('y)
        min_y_component = NumericActionComponent(0)
        max_y_component = self.dimensions[1]
        y_component_descriptor = ContinuousComponentDescriptor(min_y_component, max_y_component)

        action_descriptors.add_descriptor(y_component_name, y_component_descriptor)
