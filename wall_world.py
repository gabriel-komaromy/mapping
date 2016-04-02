from sets import Set


class World(object):
    def __init__(self):
        self.walls = Set()

    def add_wall(self, wall):
        # This lets walls intersect but I'm ok with that
        self.walls.add(wall)

        
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


class Wall(LineSegment):
    pass


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
