
# todo move to polygon_utils

# http://www.ariel.com.au/a/python-point-int-poly.html
# determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.

def point_inside_polygon(x,y,poly):
    n = len(poly)
    inside =False
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside


def find_plot(x, y, rectangles):
    for i,p in rectangles.items():
        if point_inside_polygon(x,y,p):
            return i
    return -100 # so I am sure to recognize it even if I add one to make it 1-based and then forget.

def find_plots(x, y, rectangles):
    return [find_plot(xi, yi, rectangles) for xi, yi in zip(x,y)]
