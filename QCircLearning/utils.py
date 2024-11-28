import numpy as np

def data_augment(data, n_points, shift):
    new_data = [data]
    for _ in range(n_points):
        new_data.append(data + np.random.choice([-1, 0, 1], len(data)) * shift)
    return new_data