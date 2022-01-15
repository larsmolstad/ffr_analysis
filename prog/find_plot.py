import polygon_utils_old
import polygon_utils

def find_plot(df_row, rectangles):
    # In the list or dict rectangles, every element rectangles[i] is either
    # a) a list of points representing corners of a polygon,
    # b) a Polygon object, or
    # c) a function returning True iff df_row belongs to plot i.
    x = df_row.x
    y = df_row.y
    for i, p in rectangles.items():
        if callable(p):
            if p(df_row):
                return i
        elif isinstance(p, polygon_utils.Polygon):
            if p.contains(x,y):
                return i
        else:
            if polygon_utils_old.point_inside_polygon(x, y, p):
                return i
    # so I am sure to recognize it even if I add one to make it
    # 1-based and then forget:
    return -100


def find_plots(df, rectangles):
    return [find_plot(df.loc[i], rectangles) for i in df.index]

