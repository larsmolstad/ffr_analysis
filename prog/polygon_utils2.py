"""Utilities to make, plot, move, rotate and divide rectangles and
 other 4-polygons (maybe other polygons also).

Used for defining and plotting the geometry of the plots in E22.

A 4-gon (for example a rectangle) is internally represented by the corners as
[[x1, x2, x3, x4], [y1, y2, y3, y4]]

Example:
from polygon_utils2 import Polygon, plot_rectangles
p = Polygon([0,1,1,0], [10,10,14,11])
plt.ion()
plt.cla()
p.rotate(.3, 0)
p.plot(color="green")
plt.scatter(*p.points()[0])
plt.scatter(*p.points()[1], marker='s')
d = p.divide_rectangle(3, gaps=(0,.1,0), other_way=True)
plot_rectangles(d, "0 1 2 3 4 5 6 7 8".split())
for r in d:
    plt.scatter(*r.points()[0])
    plt.scatter(*r.points()[1], marker='s')

def test_point(x,y):
    plt.scatter(x,y)
    for (i, r) in enumerate(d):
        if r.contains(x,y):
            print(i)

test_point(.3, 10.5)

plt.cla()
grid = p.grid_rectangle(3,4)
plot_rectangles(grid, range(12))
"""

from plotting_compat import plt
import numpy as np

# todo maybe make Rectangle subclass of Polygon, but this
# also works.
class Polygon(object):
    """
    These are equivalent and make a rectangle with width
    1 and height 2 with lower left corner in (0,0).
    p = Polygon([0,1,1,0], [0,0,2,2])
    p = Polygon([(0,0), (1,0), (1,2), (0,2)])
    p = Polygon(0, 0, W=1, L=1)
    """
    def __init__(self, x, y=None, W=None, L=None):
        if isinstance(x, (int, float)):
            assert(isinstance(y, (int, float)))
            self._make_rectangle(x, y, W, L)
        elif isinstance(x[0], (tuple, list, np.ndarray)):
            self.x = np.array([x[0] for x in x])
            self.y = np.array([x[1] for x in x])
        else:
            self.x = np.array(x)*1.0
            self.y = np.array(y)*1.0
        assert(len(self.x) == len(self.y))
        
    def points(self):
        return [ np.array((self.x[i], self.y[i])) 
                 for i in range(len(self.x)) ]
    
    def copy(self):
        return Polygon(self.x, self.y)
    
    def rotate(self, angle, about=0):
        """about is the point the polygon is rotated about, and can be 1) a
        pair [x,y] of coordinates, 2) an integer (zero based) representing
        which coner to rotate about, or 3) 'center' or 'c'
        
        example: 
        
        p = Polygon([5, 10 , 10, 5], [0, 0, 1, 1]], np.pi/4, 1) )
        % rotates 45 degrees about (10, 0) 
        r = p.copy().rotate(np.pi/4, 1)
        % rotates 45 degrees about center point
        r = p.copy().rotate(np.pi/4, 'c')
    """
        x, y = self.x, self.y
        if about in ('center', 'c'):
            about = (x.mean(), y.mean())
        elif isinstance(about, int):
            about = (x[about], y[about])
        x -= about[0]
        y -= about[1]
        s = np.sin(angle)
        c = np.cos(angle)
        x1 = x * c - y * s + about[0]
        y1 = x * s + y * c + about[1]
        self.x = x1
        self.y = y1
        return self

    def move(self, x, y):
        self.x += x
        self.y += y
        return self
    

    def _make_rectangle(self, x0, y0, W, L):
        self.x = np.array([0, W, W, 0]) + x0
        self. y = np.array([0, 0, L, L]) + y0
        
    def grid_rectangle(self, m, n, mgaps=(0,0,0), ngaps=(0,0,0)):
        """Divides the rectangle
          p3 --- p2     
          |       |
          p0 --- p1
      
        (possibly rotated) into m x n equal rectangles (m rows and n columns)
        numbered like so, if m=2 and n=3:
        
           r1  r2  r3
           r4  r5  r6

        See divide_rectangle for description of gaps"""
        rectlist = self.divide_rectangle(m, gaps=mgaps, other_way=True)
        rectlists = [r.divide_rectangle(n, gaps=ngaps) for r in rectlist]
        return sum(rectlists, [])
    
    def divide_rectangle(self, n, other_way=False, gaps=(0,0,0)):
        """Divides the rectangle
    
          p3 --- p2     
          |       |
          p0 --- p1
      
        (possibly rotated) into n equal rectangles. If other_way is
        false, new points are inserted between p1 and p2, and p0 and
        p3.  If gaps are not all zero, there will be gaps like so (for
        n = 2, other_way=False):
            
            p3     p2
              gaps[2]
            r3 --- r2
            |       |
            r0 --- r1 
              gaps[1]
            q3 --- q2 
            |       |
            q0 --- q1 
              gaps[0]
            p0     p1  
        
        gaps[1] is repeated if n>2.
        Actually, it doesn't need to be a rectangle, just a 4-gon
        """
        
        p = self.points()
        if other_way:
            p = [p[-1], p[0], p[1], p[2]]
        unitv1 = (p[1] - p[0])/np.linalg.norm(p[1]-p[0])
        unitv2 = (p[2] - p[3])/np.linalg.norm(p[2]-p[3])
        p[0] += unitv1*gaps[0]
        p[3] += unitv2*gaps[0]
        p[1] -= unitv1*gaps[0]
        p[2] -= unitv2*gaps[0]
        L1 = np.linalg.norm(p[1]-p[0])
        l1 = (L1+gaps[1])/n
        L2 = np.linalg.norm(p[2]-p[3])
        l2 = (L2+gaps[1])/n
        p0s = np.array([p[0] + unitv1*l1*i for i in range(n)])
        p1s = p0s + (l1-gaps[1])*unitv1
        p3s = np.array([p[3] + unitv2*l2*i for i in range(n)])
        p2s = p3s + (l2-gaps[1])*unitv2
        print(p0s)
        print(p1s)
        print(p2s)
        print(p3s)
        if other_way:
            p0s, p1s, p2s, p3s = p1s, p2s, p3s, p0s
        return [Polygon((p0s[i], p1s[i], p2s[i], p3s[i])) for i in range(n)]        
        
            
    def plot(self, text=None, textkwargs= {}, **kwargs):
        x, y = self.x, self.y
        #todo callable .... sometimes before I made this a class I called 
        # plot_rectangle on  something which wasn't rectangle, but a function
        if not 'color' in kwargs:
            kwargs['color'] = 'k'
        plt.plot(list(x) + [x[0]], list(y) + [y[0]], **kwargs)
        if not 'fontsize' in textkwargs.keys():
            textkwargs['fontsize'] = 8
        if not(text is None):
            x, y = self.midpoint()
            plt.text(x, y, text, **textkwargs)

    def midpoint(self):
        return self.x.mean(), self.y.mean()

    def __repr__(self):
        return "Polygon with\n x = {}\n y = {}".format(self.x, self.y)
    

    def contains(self, x, y):
        """ return True if a point is inside the polygon, otherwise False"""
    # http://www.ariel.com.au/a/python-point-int-poly.html
    # determine if a point is inside a given polygon or not
        poly = self.points()
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    


def plot_rectangles(rectangles, names=True, textkwargs={}, **kwargs):
    """rectangles can be a dict or a list of rectangles. If rectangles is
a dict and names==True, the keys are usesd as names. names may also be
a list"""
    assert(isinstance(rectangles, (dict, list)))
    if isinstance(rectangles, dict):
        pairs = [(key, rectangles[key]) for key in list(rectangles)]
        rectangles = [x[1] for x in pairs]
        if names is True:
            names = [x[0] for x in pairs]
    else:
        if names is True:
            names = range(1, len(rectangles)+1)
    for i, r in enumerate(rectangles):
        if callable(r): # for the pot experiment, where we don't have rectangles, but a list of functions. untested todo.
            continue
        r.plot(text=None if not names else names[i],
               textkwargs=textkwargs,
               **kwargs)
    
# fast enough so far:
def find_polygon(x, y, polygons):
    """ return index of the first polygon (in a list of polygons) containing (x,y).
    return -1 if (x,y) is not inside any of the polygons."""
    for (i, r) in enumerate(polygons):
        if r.contains(x, y):
            return i
    return -1



