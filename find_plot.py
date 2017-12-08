
# todo move to polygon_utils

# http://www.ariel.com.au/a/python-point-int-poly.html
# determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.


def point_inside_polygon(x, y, poly):
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


def find_plot(df_row, rectangles):
    # In the list rectangles, every element rectangles[i] is either
    # a) a list of points representing corners of a polygon, or
    # b) a function returning True iff df_row belongs to plot i.
    x = df_row.x
    y = df_row.y
    for i, p in rectangles.items():
        if callable(p):
            if p(df_row):
                return i
        else:
            if point_inside_polygon(x, y, p):
                return i
    # so I am sure to recognize it even if I add one to make it
    # 1-based and then forget:
    return -100


def find_plots(df, rectangles):
    return [find_plot(df.loc[i], rectangles) for i in df.index]

