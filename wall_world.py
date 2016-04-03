from sets import Set
import math

from actions import ComponentName
from observations import NumericFeatureValue
from observations import Observation
from agents import SingleAgentID


class LineSegment(object):
    def __init__(self, endpoints):
        self.endpoints = endpoints


    # bryceboe.com/2006/10/23/line-segment-intersection-algorithm
    def intersects(self, other):
        A, B = self.endpoints
        C, D = other.endpoints
        def ccw(A, B, C):
            return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    # stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
    def line_intersection(self, other):
        A, B = self.endpoints
        C, D = other.endpoints

        xdiff = (A.x - B.x, C.x - D.x)
        ydiff = (A.y - B.y, C.y - D.y)

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)

        if div == 0:
            raise ValueError("Lines do not intersect")

        d = (det((A.x, A.y), (B.x, B.y)), det((C.x, C.y), (D.x, D.y)))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Point(x, y)

    def slope(self):
        try:
            return (self.endpoints[1].y - self.endpoints[0].y) / (self.endpoints[1].x - self.endpoints[0].x)
        except:
            raise ValueError()

    def __str__(self):
        return str(self.endpoints[0]) + ", " + str(self.endpoints[1])


class Wall(LineSegment):
    def __init__(self, endpoints):
        super(Wall, self).__init__(endpoints)

class Trajectory(LineSegment):
    def __init__(self, endpoints):
        super(Trajectory, self).__init__(endpoints)


class Point(object):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"


class Direction(object):
    def __init__(self, components):
        self.x_component = components[0]
        self.y_component = components[1]

    def __str__(self):
        return "<" + str(self.x_component) + ", " + str(self.y_component) + ">"


class World(object):
    directions = {
        'north': Direction((0, 1)),
        'east': Direction((1, 0)),
        'south': Direction((0, -1)),
        'west': Direction((-1, 0)),
        }
    component_names = {
        'x': ComponentName('x'),
        'y': ComponentName('y'),
        }

    def __init__(self, dimensions, feature_names, robot_position):
        self.dimensions = dimensions
        self.feature_names = feature_names
        self.add_robot(Point(robot_position[0], robot_position[1]))
        self.walls = Set()
        self.agent = SingleAgentID()
        bottom_left = Point(1, 1)
        top_left = Point(1, dimensions[1] - 1)
        top_right = Point(dimensions[0] - 1, dimensions[1] - 1)
        bottom_right = Point(dimensions[0] - 1, 1)
        self.add_wall(Wall((bottom_left, top_left)))
        self.add_wall(Wall((top_left, top_right)))
        self.add_wall(Wall((top_right, bottom_right)))
        self.add_wall(Wall((bottom_right, bottom_left)))

        self.read_walls_from_file('wall_list.txt')

    def read_walls_from_file(self, file_name):
        with open(file_name, 'r') as f:
            walls_list = f.readlines()

        split_coordinates = [wall.split(',') for wall in walls_list]
        wall_coordinates = []
        for wall in split_coordinates:
            wall_coordinates.append([float(coord.strip()) + 1 for coord in wall])
        for wall in wall_coordinates:
            start = Point(wall[0], wall[1])
            end = Point(wall[2], wall[3])
            self.add_wall(Wall((start, end)))

    def add_wall(self, wall):
        # This lets walls intersect but I'm ok with that
        assert self.in_boundaries(wall.endpoints[0])
        assert self.in_boundaries(wall.endpoints[1])
        self.walls.add(wall)

    def add_robot(self, location):
        assert self.in_boundaries(location)
        self.robot_location = location

    def make_observation(self):
        observation = Observation()
        observation.add_feature(self.feature_names['x'], NumericFeatureValue(self.robot_location.x))
        observation.add_feature(self.feature_names['y'], NumericFeatureValue(self.robot_location.y))
        for dir in self.directions.keys():
            observation.add_feature(self.feature_names[dir], NumericFeatureValue(self.distance_in_direction(self.directions[dir])))
        return observation

    def initial_state(self):
        return self.make_observation()

    def update(self, action_map, term_signal):
        action = action_map.get_action(self.agent)
        new_x = action.get_component(self.component_names['x']).action_value
        new_y = action.get_component(self.component_names['y']).action_value
        self.robot_location = self.move_robot(Point(new_x, new_y))
        return self.make_observation()

    def move_robot(self, destination):
        assert self.robot_location is not None
        assert self.in_boundaries(destination)
        trajectory = Trajectory((self.robot_location, destination))
        intersections = self.intersections(self.walls, trajectory)
        if len(intersections) == 0:
            new_location = destination
        else:
            # if the movement would run into a wall, end the movement a short distance away from the wall
            collision_distance = 0.1
            closest_intersection = self.closest_intersection(intersections)
            try:
                movement_direction = trajectory.slope()
                correction = -1 * movement_direction
                if correction == 0:
                    # horizontal line so slope is 0
                    direction = self.single_coordinate_direction(trajectory.endpoints[1].x, trajectory.endpoints[0].x)
                    x_offset = collision_distance * direction

                    y_offset = 0

                else:
                    x_offset = collision_distance / correction
                    y_offset = collision_distance * correction
            except:
                # vertical line so slope is undefined
                x_offset = 0

                direction = self.single_coordinate_direction(trajectory.endpoints[1].y, trajectory.endpoints[0].y)
                y_offset = collision_distance * direction

            new_location = Point(closest_intersection.x + x_offset, closest_intersection.y + y_offset)
        return new_location

    """Handles cases of horizontal/vertical movement collisions"""
    def single_coordinate_direction(self, destination, origin):
        if destination > origin:
            return -1
        else:
            return 1

    def in_boundaries(self, location):
        x = location.x
        y = location.y
        return 0 <= x <= self.dimensions[0] and 0 <= y <= self.dimensions[1]

    def intersections(self, walls, trajectory):
        intersections = Set()
        for wall in walls:
            if wall.intersects(trajectory):
                intersections.add(wall.line_intersection(trajectory))
        return intersections

    def distance_in_direction(self, direction):
        traj = Trajectory((self.robot_location, Point(direction.x_component * self.dimensions[0] * 2, direction.y_component * self.dimensions[1] * 2)))
        intersections = self.intersections(self.walls, traj)
        assert len(intersections) > 0

        return self.robot_location.distance(self.closest_intersection(intersections))

    def closest_intersection(self, intersections):
        return min(intersections, key=lambda point: self.robot_location.distance(point))
