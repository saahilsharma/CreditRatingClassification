import json
import numpy as np


def load_json(json_file):
    with open(json_file, "r") as fin:
        data = json.load(fin)
    
    return data


def evaluate(predicted_y, y):
    error = predicted_y - y
    mse = np.average(error ** 2)

    return mse


def count_nonzero_coef(coef):
    cnt = 0

    for i in coef:
        # check float variable equals to zero
        if abs(i) >= 1e-9:
            cnt += 1

    return cnt