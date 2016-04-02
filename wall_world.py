from sets import Set
import math


class World(object):
    def __init__(self, boundaries):
        # boundaries are an x length and a y length
        # position indexing starts at 0 in the bottom left
        self.boundaries = boundaries
        self.walls = Set()

    def add_wall(self, wall):
        # This lets walls intersect but I'm ok with that
        assert in_boundaries(wall.endpoints[0])
        assert in_boundaries(wall.endpoints[1])
        self.walls.add(wall)

    def add_robot(self, location):
        assert self.in_boundaries(location)
        self.robot_location = location

    def move_robot(self, destination):
        assert self.robot_location is not None
        assert self.in_boundaries(destination)
        trajectory = Trajectory((self.robot_location, destination))
        intersections = Set()
        for wall in self.walls:
            if wall.intersects(trajectory):
                intersections.add(wall.line_intersection(trajectory))
        if len(intersections) == 0:
            self.robot_location = destination
        else:
            # if the movement would run into a wall, end the movement a short distance away from the wall
            collision_distance = 0.1
            closest_intersection = min(intersections, key=lambda point: self.robot_location.distance(point))
            try:
                movement_direction = trajectory.slope()
                correction = -1 * movement_direction
                if correction == 0:
                    # horizontal line so slope is 0
                    direction = self.slopeless_movement_direction(trajectory.endpoints[1].x, trajectory.endpoints[0].x)
                    x_offset = collision_distance * direction
                    y_offset = 0
                else:
                    x_offset = collision_distance / correction
                    y_offset = collision_distance * correction
            except:
                # vertical line so slope is undefined
                x_offset = 0

                direction = self.slopeless_movement_direction(trajectory.endpoints[1].y, trajectory.endpoints[0].y)

                y_offset = collision_distance * direction
            new_location = Point(closest_intersection.x + x_offset, closest_intersection.y + y_offset)
            self.robot_location = new_location
        print self.robot_location

    """Handles cases of horizontal/vertical movement collisions"""
    def slopeless_movement_direction(self, destination, origin):
        if destination > origin:
            return -1
        else:
            return 1

    def in_boundaries(self, location):
        x = location.x
        y = location.y
        return 0 <= x <= self.boundaries[0] and 0 <= y <= self.boundaries[1]

        
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
