import numpy as np
upper_bucket_diameter = 23.5
lower_bucket_diameter = 21.5
bucket_height = 21.5

raw_data = {(1, 'right'): [5.5, 5.9, 5.6],
            (1, 'left'): [5.2, 4.9, 5.3],
            (2, 'right'): [7.5, 6.9, 7.6],
            (2, 'left'): [6.2, 6.4, 7],
            (3, 'right'): [6, 5.4, 5.2],
            (3, 'left'): [5, 5.4, 5.5],
            (4, 'right'): [6.1, 6.6, 7],
            (4, 'left'): [5.3, 6, 5.7],
            (5, 'right'): [6.3, 6, 6],
            (5, 'left'): [6.5, 6.2, 6.3],
            (6, 'right'): [5.7, 5.3, 4.9],
            (6, 'left'): [5.8, 5.9, 5.5],
            (7, 'right'): [6, 6, 5.5],
            (7, 'left'): [5.4, 5.3, 5.3],
            (8, 'right'): [5, 5, 5],
            (8, 'left'): [6, 5.8, 6.2],
            (9, 'right'): [7, 6.9, 6.5],
            (9, 'left'): [7, 7, 6.4],
            (10, 'right'): [5.9, 6, 5.4],
            (10, 'left'): [4.4, 4.3, 4.4],
            (11, 'right'): [4.9, 5, 4.8],
            (11, 'left'): [4.5, 5, 5],
            (12, 'right'): [5.6, 5.9, 6],
            (12, 'left'): [4.4, 5, 5.4],
            (13, 'right'): [3.9, 3.5, 2.9],
            (13, 'left'): [4.4, 4.4, 4.3],
            (14, 'right'): [4.1, 4.1, 4.1],
            (14, 'left'): [5, 5, 5.5],
            (15, 'right'): [5.4, 5.5, 4],
            (15, 'left'): [5, 5, 5],
            (16, 'right'): [5.5, 5.2, 5.2],
            (16, 'left'): [4.5, 4.5, 5.4],
            (17, 'right'): [5.5, 5.7, 6.2],
            (17, 'left'): [4.3, 4.4, 5],
            (18, 'right'): [4.2, 4, 3.9],
            (18, 'left'): [4.1, 4.2, 4.2],
            (19, 'right'): [5.2, 5.5, 5.1],
            (19, 'left'): [6, 5.4, 4.5],
            (20, 'right'): [5.5, 5.5, 6],
            (20, 'left'): [5, 5.6, 5.4],
            (21, 'right'): [5.5, 5.5, 5.6],
            (21, 'left'): [5.7, 5.7, 5.9],
            (22, 'right'): [7.2, 6.3, 6.6],
            (22, 'left'): [8, 7.5, 7.6],
            (23, 'right'): [7, 6.5, 6.6],
            (23, 'left'): [5.5, 5.9, 5.8],
            (24, 'right'): [6, 6.6, 6.7],
            (24, 'left'): [6.5, 6.1, 6]}

top_gaps = {key: np.mean(x) for key, x in raw_data.items()}


def volume(gap):
    h = bucket_height - gap
    dtop = (upper_bucket_diameter * gap +
            lower_bucket_diameter * h) / bucket_height
    rtop = dtop / 2
    rbot = lower_bucket_diameter / 2
    return np.pi / 3 * h * (rtop**3 - rbot**3) / (rtop - rbot)


soil_volumes = {key: volume(x) for key, x in top_gaps.items()}
