"""
Helper functions.
"""
import numpy as np
import scipy.stats
from PIL import Image


def get_teacher_signal(width, height, xmin, ymin, xmax, ymax, sigma=4, downsample=4):
    """Creates teacher signal for the image."""
    signal = np.zeros((height, width))

    c_x = xmin + int((xmax - xmin) / 2)
    c_y = ymin + int((ymax - ymin) / 2)

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            signal[y, x] = scipy.stats.multivariate_normal.pdf([y, x], [c_y, c_x], [sigma, sigma])

    # downsample
    signal = signal[::downsample, ::downsample]
    return signal


def create_dataset(csv_file, obj=None, sep=',', sigma=4, downsample=4, verbose=True):
    """Creates dataset from csv_file."""

    dataset = []
    count = 0

    with open(csv_file, "r") as f:
        csv_lines = f.readlines()[1:]

    for line in csv_lines:
        cols = line.strip("\n").split(sep)
        if obj is not None:
            if cols[3] == obj:
                if verbose:
                    print("Preparing sample #", count)
                    count += 1

                file_name = cols[0]
                width = int(cols[1])
                height = int(cols[2])
                xmin = int(cols[4])
                ymin = int(cols[5])
                xmax = int(cols[6])
                ymax = int(cols[7])

                # create signal
                signal = get_teacher_signal(width, height, xmin, ymin, xmax, ymax, sigma=sigma, downsample=downsample)

                # extract image array to memory
                img = np.array(Image.open("data/" + file_name))

                dataset.append([img, signal])

    return dataset


ds = create_dataset("data/data.csv", 'ball')
print(len(ds))
