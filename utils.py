import numpy as np


def calculate_min_max(cluster):
    X=[]
    Y=[]
    for i in cluster:
        for j in i:
            X.append(j[0])

    for i in cluster:
        for j in i:
            Y.append(j[1])

    min_X = min(X,default=0)
    max_X = max(X,default=0)
    min_Y = min(Y,default=0)
    max_Y = max(Y,default=0)
    return min_X,max_X,min_Y,max_Y


def image_cropping( img):
    crop_img = img[:, 1749:3240]
    crop_region = np.array((1749, 3240))
    ignore_region = np.array([
        (1749, 1865),
        (2620, 2653)
    ])
    ignore_region_adjusted = ignore_region - crop_region[0]
    MARGIN = 5
    for left, right in ignore_region_adjusted:
        crop_img[:, max(0, left - MARGIN):min(img.shape[1], right + MARGIN)] = 0

    return crop_img