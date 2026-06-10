import math

from configs import configs

import numpy as np

seed = configs['seed']
tolerance = configs['tolerance']

rng = np.random.default_rng(seed)

def generate_datapoint():
    choice = rng.integers(0, 3, endpoint=True)
    metric = rng.integers(1, 100, endpoint=True)
    # arbitrary choice of distributions to mix together
    new_datapoint = 0
    if choice == 0:
        new_datapoint = rng.exponential(scale=1/metric)
    elif choice == 1:
        new_datapoint = rng.chisquare(df=metric)
    elif choice == 2:
        new_datapoint = rng.normal(loc=metric, scale=metric/10)
    else:
        new_datapoint = rng.geometric(p = 1/metric)
    return new_datapoint

def calc_mean(dataset):
    data_count = 0
    data_sum = 0
    for i in range(len(dataset)):
        data_count += 1
        data_sum += dataset[i]
    return data_sum / data_count

def calc_variance(dataset, mean):
    data_count = 0
    sum = 0
    for num in dataset:
        data_count += 1
        sum += (num - mean)**2
    return sum / data_count

def generate_dataset(count, mean, std) -> list:
    result = []

    for i in range(count):
        result.append(generate_datapoint())

    raw_mean = calc_mean(result)
    raw_std = math.sqrt(calc_variance(result, raw_mean))

    result = list(map(lambda x: (x - raw_mean) / raw_std, result))
    result = list(map(lambda x: x * std + mean, result))

    return result

if __name__ == "__main__":
    import json
    test_used = json.load(open('test_cases.json', "r"))[configs["test_used"]-1]

    # generate 2 datasets
    stats_1, stats_2 = test_used[0], test_used[1]
    X_1 = generate_dataset(stats_1['n'], stats_1['mean'], stats_1['std'])
    X_2 = generate_dataset(stats_2['n'], stats_2['mean'], stats_2['std'])