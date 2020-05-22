def get_previous(vector, samples):
    for current in range(0, len(vector), 2):
        yield [vector[current - i] for i in range(samples)]


def split(vector_xy):
    return [x for x, y in vector_xy], [y for x, y in vector_xy]


def normalise(vector_xy):
    v_x, v_y = split(vector_xy)
    x_max, x_min, y_max, y_min = max(v_x), min(v_x), max(v_y), min(v_y)
    return [[(x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for x, y in vector_xy], (x_min, x_max), (y_min, y_max)


def normalise_from_to(vector_xy, from_x, from_y, to_x, to_y):
    return [[(x - from_x[0]) * (to_x[1] - to_x[0]) / (from_x[1] - from_x[0]) + to_x[0],
             (y - from_y[0]) * (to_y[1] - to_y[0]) / (from_y[1] - from_y[0]) + to_y[0]] for x, y in vector_xy]


def reduce(vector_xy):
    new_v = []
    for x, y in vector_xy:
        new_v.append(x)
        new_v.append(y)
    return new_v
