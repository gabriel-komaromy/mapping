from sets import Set
import math


class World(object):
    def __init__(self):
        self.walls = Set()

    def add_wall(self, wall):
        # This lets walls intersect but I'm ok with that
        self.walls.add(wall)

    def add_robot(self, location):
        self.robot_location = location

    def move_robot(self, destination):
        assert self.robot_location is not None
        trajectory = Trajectory((self.robot_location, destination))
        intersections = Set()
        for wall in self.walls:
            if wall.intersects(trajectory):
                intersections.add(wall.line_intersection(trajectory))
        closest_intersection = min(intersections, key=lambda point: self.robot_location.distance(point))

        
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
