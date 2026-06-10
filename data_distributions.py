import json

data_stats = []

def stats_generator(n1, mean1, std1, n2, mean2, std2):
    test = []
    A = {
        "n": n1,
        "mean": mean1,
        "std": std1
    }
    B = {
        "n": n2,
        "mean": mean2,
        "std": std2
    }
    test.append(A)
    test.append(B)
    return test

test_cases = [
    # Test Case 1: Same mean, Same std
    (100, 1000, 100, 100, 1000, 100),
    # Test Case 2: Different (small) mean, Same std
    (90, 500, 50, 90, 480, 50),
    #  Test Case 3: Same mean, Different (big) std
    (50, 500, 5, 50, 500, 50),
    # Test Case 4: Large n but small m, Different (small) mean, Different (big) std
    (10000, 500, 50, 7, 480, 5)
]

for case in test_cases:
    data_stats.append(stats_generator(*case))

with open("test_cases.json", "w") as file:
    json.dump(data_stats, file)
