# python3
# motion planning demos
# LI Yunsheng
# 2016.01.25


class Point(object):
    """
    2D Point in Plane
    """
    _x = 0.
    _y = 0. 
    _parent = None
    _index = -1 


    def __init__(self, x=0., y=0.):
        self._x = x
        self._y = y
        self._parent = None
        self._index = -1


    @property
    def x(self):
        return self._x


    @property
    def y(self):
        return self._y


    @property
    def parent(self):
        return self._parent
    

    @parent.setter
    def parent(self, p):
        self._parent = p


    @property
    def index(self):
        return self._index


    @index.setter
    def index(self, i):
        self._index = i
    


    def __sub__(self, other):
        return self._x - other.x, self._y - other.y 


    @classmethod
    def translation(cls, ref_point, dx=0., dy=0.):
        return cls(ref_point.x + dx, ref_point.y + dy)


    def distance(self, other):
        from math import sqrt
        return sqrt((self._x - other.x)**2 + (self._y - other.y)**2)


    def intersection(self, line):
        """
        Point on Line
        """
        res = False 
        eps = 1.e-6
        if line.length < eps:
            if self.distance(line.P1) < eps and self.distance(line.P2) < eps:
                res = True
            return res
        if abs(Line(line.P1, self) @ Line(self, line.P2)) < eps:
            try:
                t = (self._x - line.P1.x) / (line.P2.x - line.P1.x)
            except ZeroDivisionError:
                t = (self._y - line.P1.y) / (line.P2.y - line.P1.y)
            finally:
                if 0 <= t <= 1:
                    res = True
        return res


    def at_left_of(self, line):
        """
        check if point stay at left of line 
        """
        return Line(line.P1, self) @ Line(self, line.P2) > 0.




class Line(object):
    """
    Line in Plane connect two points
    """
    _P1 = Point()
    _P2 = Point()
    _dx, _dy = _P2 - _P1


    def __init__(self, p1=Point(), p2=Point()):
        from math import sqrt
        self._P1 = p1
        self._P2 = p2 
        self._dx, self._dy = p2 - p1
        self._length = sqrt(self._dx**2 + self._dy**2)


    @property
    def P1(self):
        return self._P1


    @property
    def P2(self):
        return self._P2


    @property
    def dx(self):
        return self._dx


    @property
    def dy(self):
        return self._dy


    @property
    def length(self):
        return self._length


    @classmethod
    def vector(cls, dx=0., dy=0., ref_point=Point()): 
        P1 = ref_point
        P2 = Point.translation(dx, dy, ref_point=P1)
        return cls(P1, P2)


    def __lt__(self, other):
        return self._length < other.length


    def relative_to(self, other):
        """
        relative coordinates to other line
        """
        eps = 1.e-6
        if other.length > eps:
            from math import asin, cos, sin
            theta = asin(other.dy / other.length)
            c_t, s_t = cos(theta), sin(theta)
            P1 = Point((other.P1.x-self.P1.x)*c_t+(other.P1.y-self.P1.y)*s_t, -(other.P1.x-self.P1.x)*s_t+(other.P1.y-self.P1.y)*c_t)
            P2 = Point((other.P1.x-self.P2.x)*c_t+(other.P1.y-self.P2.y)*s_t, -(other.P1.x-self.P2.x)*s_t+(other.P1.y-self.P2.y)*c_t)
            return Line(P1, P2)
        else:
            return None


    def __mul__(self, other):
        """
        Inner Product
        """
        return self._dx * other.dx + self._dy * other.dy


    # @classmethod
    def __matmul__(self, other):
        """
        Cross Product: @
        """
        return self._dx * other.dy - self._dy * other.dx


    def intersection(self, other):
        """
        Line intersect with another Line 
        """
        res = False
        ref_line = self.relative_to(other)
        if ref_line is not None:
            try:
                x = (ref_line.P1.x*ref_line.P2.y - ref_line.P2.x*ref_line.P1.y) / (ref_line.P2.y - ref_line.P1.y)
            except:
                x = -1.
            finally:
                if not x < 0. and not x > other.length:
                    res = True 
        return res



class  Polygon(object):
    """
    2D Polygon
    """
    Point_List = []
    Line_List = []


    def __init__(self, point_list):
        """
        points is point list of polygon, the points is counterclockwise stored in Point_List
        """
        self.Point_List = point_list 
        self.Line_List = [Line(point_list[i], point_list[i+1]) for i in range(len(point_list) - 1)]


    def patch(self, color='green'):
        """
        argument for plot/ax.add_patch()
        """
        from matplotlib.path import Path
        import matplotlib.patches as patches
        codes = [Path.MOVETO]
        codes.extend([Path.LINETO for i in range(len(self.Line_List))])
        codes.append(Path.CLOSEPOLY)

        verts = [(p.x, p.y) for p in self.Point_List]
        verts.append(verts[0])

        path = Path(verts, codes)

        return patches(path, facecolor=color)


    def inner_point(self, point=Point()):
        res = True
        for line in self.Line_List:
            if not point.at_left_of(line):
                res = False
                break 
        return res 


    def intersection(self, L=Line()):
        """
        check if line is collision with polygon
        """
        res = False
        for line in self.Line_List:
            if L.intersection(line):
                res = True
                break 
        if not res:
            if self.inner_point(L.P1):
                res = True 
        return res



class Obstacle(object):
    """
    list of polygons
    """
    Polygon_List = []

    def __init__(self, p_list=[]):
        self.Polygon_List = p_list 


    def intersection(self, line):
        res = False
        for polygon in self.Polygon_List:
            res = polygon.intersection(line)
            if res:
                break
        return res


class Workspace(object):
    def __init__(self, length=100., width=100., obstacle=Obstacle()):
        self._length = length 
        self._width = width
        self._obstacle = obstacle
        self._boundary = None


    @property
    def length(self):
        return self._length
    

    @property
    def width(self):
        return self._width
    

    @property
    def obstacle(self):
        return self._obstacle


    @obstacle.setter
    def obstacle(self, obstacle):
        self._obstacle = obstacle


    def add_obstacle(self, polygon):
        self._obstacle.Polygon_List.append(polygon)


    @property
    def boundary(self):
        return self._boundary


    @boundary.setter
    def boundary(self, polygon):
        self._boundary = polygon
    
    


def NearNeighbors(sample, point_list, k=1):
    if k >= len(point_list):
        return point_list
    else:
        index = set()
        line_list = []
        tmp_index = set((-1,))
        import random
        times = 0 
        while times < 3:
            tmp_index = set([random.randrange(len(point_list)) for _ in range(k)])
            tmp_index -= index
            if not tmp_index:
                continue
            line_list.extend([Line(point_list[i], sample) for i in tmp_index])
            line_list.sort()
            line_list = line_list[0:min(k, len(line_list))]
            tmp_index = index 
            index = set([line.P1.index for line in line_list])
            if not (tmp_index - index):
                times += 1
            if len(index) < k:
                times = 0 
        return [point_list[i] for i in index]





def main():

    import random

    ws = Workspace()
    ws.add_obstacle(Polygon([Point(11,73), Point(45,73), Point(45,82), Point(11,82)]))
    ws.add_obstacle(Polygon([Point(9,12), Point(34,12), Point(34,36), Point(9,36)]))
    ws.add_obstacle(Polygon([Point(72,8), Point(93,8), Point(93,29), Point(72,29)]))
    ws.add_obstacle(Polygon([Point(55,38), Point(77,38), Point(77,94), Point(55,94)]))
    ws.add_obstacle(Polygon([Point(77,66), Point(94,66), Point(94,73), Point(77,73)]))

    start = Point(50.,50.)
    start.index = 0

    goal_region = Polygon([Point(83,83), Point(97,83), Point(97,97), Point(83,97)])

    RRT = [start]

    reach = False

    while not reach:
        sample = Point(random.uniform(0., ws.length), random.uniform(0., ws.width))
        nearest = NearNeighbors(sample, RRT, 1)[0]
        line = Line(nearest, sample)
        if not ws.obstacle.intersection(line):
            # 碰撞检查： 线与多边形相交，未考虑在多边形内部的情况
            line.P2.index = len(RRT)
            line.P2.parent = line.P1
            RRT.append(line.P2)
            if goal_region.inner_point(line.P2):
                reach = True




if __name__ == '__main__':
    main()