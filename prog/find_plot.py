import polygon_utils


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
            if polygon_utils.point_inside_polygon(x, y, p):
                return i
    # so I am sure to recognize it even if I add one to make it
    # 1-based and then forget:
    return -100


def find_plots(df, rectangles):
    return [find_plot(df.loc[i], rectangles) for i in df.index]

