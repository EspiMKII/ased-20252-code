import numpy as np
import math
from scipy import stats

######################################################################
# Configs

# the answer to the ultimate question of life, the universe, and everything.
SEED = 42

N_1 = 1000
MEAN_1 = 980
STD_1 = 56

N_2 = 1000
MEAN_2 = 1000
STD_2 = 55

TOLERANCE = 1e-6

USE_U_FORMULA = True
VERBOSE = True

CONFIDENCE_LEVEL = 0.95
######################################################################
# Data Generation

# the new way of using np.random is really nice since the seed makes it reproducible
rng = np.random.default_rng(SEED)
X1, X2 = [], []
def generate_datapoint():
    choice = rng.integers(0, 3, endpoint=True)
    metric = rng.integers(0, 100, endpoint=True)
    # arbitrary choice of distributions to mix together
    new_datapoint = 0
    if choice == 0:
        new_datapoint = rng.exponential(scale=1/metric)
    elif choice == 1:
        new_datapoint = rng.chisquare(df=metric)
    elif choice == 2:
        new_datapoint = rng.normal(loc=metric, scale=metric/10)
    else:
        new_datapoint = rng.geometric(0.5)
    return new_datapoint
for i in range(N_1):
    new_datapoint = generate_datapoint()
    X1.append(new_datapoint)
for i in range(N_2):
    new_datapoint = generate_datapoint()
    X2.append(new_datapoint)

def mean(dataset):
    data_count = 0
    sum = 0
    for num in dataset:
        data_count += 1
        sum += num
    return sum / data_count
def variance(dataset, mean):
    data_count = 0
    sum = 0
    for num in dataset:
        data_count += 1
        sum += (num - mean)**2
    return sum / data_count
raw_mean_1 = mean(X1)
raw_sigma_1 = math.sqrt(variance(X1, raw_mean_1))
raw_mean_2 = mean(X2)
raw_sigma_2 = math.sqrt(variance(X2, raw_mean_2))

X1_normalized = list(map(lambda x: (x - raw_mean_1)/raw_sigma_1, X1))
X1 = list(map(lambda x: x * STD_1 + MEAN_1, X1_normalized))

X2_normalized = list(map(lambda x: (x - raw_mean_2)/raw_sigma_2, X2))
X2 = list(map(lambda x: x * STD_2 + MEAN_2, X2_normalized))

print(f"X1: count: {N_1}; mean: {MEAN_1}; standard deviation: {STD_1}")
print(f"X2: count: {N_2}; mean: {MEAN_2}; standard deviation: {STD_2}")
if (abs(np.mean(X1) - MEAN_1) < TOLERANCE) and \
   (abs(math.sqrt(np.var(X1)) - STD_1) < TOLERANCE) and \
   (abs(np.mean(X2) - MEAN_2) < TOLERANCE) and \
   (abs(math.sqrt(np.var(X2)) - STD_2) < TOLERANCE) and \
   VERBOSE:
    print("Verified with NumPy")
print()

print(f"Confidence level: {CONFIDENCE_LEVEL} => alpha/2: {(1-CONFIDENCE_LEVEL)/2:.3f}")
print()
######################################################################
# Processing

# Sorting & Ranking
def merge_sort(A, n, l, r):
    m = (l+r) // 2
    if l < r:
        left = merge_sort(A, n, l, m)
        right = merge_sort(A, n, m+1, r)

        l_idx, r_idx = 0, 0
        result = []
        while l_idx < len(left) and r_idx < len(right):
            if left[l_idx] < right[r_idx]:
                result.append(left[l_idx])
                l_idx += 1
            else:
                result.append(right[r_idx])
                r_idx += 1
        if l_idx < len(left):
            idx, bound, arr = l_idx, len(left), left
        else:
            idx, bound, arr = r_idx, len(right), right
        while idx < bound:
            result.append(arr[idx])
            idx += 1

        return result
    else:
        return [A[l]]

X1 = merge_sort(X1, N_1, 0, N_1-1)
X2 = merge_sort(X2, N_2, 0, N_2-1)

# We could also just call the default sort() function which would be faster
# X1.sort()
# X2.sort()

X_merged = []
idx1, idx2 = 0, 0
while idx1 < N_1 and idx2 < N_2:
    if X1[idx1] <= X2[idx2]:
        temp = (1, X1[idx1])
        idx1 += 1
    else:
        temp = (2, X2[idx2])
        idx2 += 1
    X_merged.append(temp)
group_num ,idx, remaining, bound = 0, 0, [], 0
if idx1 < N_1:
    group_num, idx, remaining, bound = 1, idx1, X1, N_1
if idx2 < N_2:
    group_num, idx, remaining, bound = 2, idx2, X2, N_2
while idx < bound:
    X_merged.append((group_num ,remaining[idx]))
    idx += 1

# if VERBOSE:
#     print("X_merged:")
#     for num in X_merged:
#         print(f"({num[0]}, {num[1]:.2f})", end = ' ')
#     print()

# Computing U score
U_1, U_2 = 0, 0
ranked_1, ranked_2 = [], []
for i in range(len(X_merged)):
    num, rank = X_merged[i], i+1
    if num[0] == 1:
        ranked_1.append(rank)
    if num[0] == 2:
        ranked_2.append(rank)

if USE_U_FORMULA:
    R_1 = sum(ranked_1)
    U_1 = N_1 * N_2 + N_1 * (N_1 + 1) * 0.5 - R_1
    R_2 = sum(ranked_2)
    U_2 = N_1 * N_2 + N_2 * (N_2 + 1) * 0.5 - R_2
else:
    for rank1 in ranked_1:
        for rank2 in ranked_2:
            if rank1 > rank2:
                U_1 += 1
            elif rank1 < rank2:
                U_2 += 1
            else: # tiebreak
                U_1 += 0.5
                U_2 += 0.5

U = min(U_1, U_2)

if VERBOSE:
    print(f"U_1: {U_1}")
    print(f"U_2: {U_2}")
    print(f"U: {U}")

# Transforming to Z score & Comparing to alpha

U_mean = N_1 * N_2 * 0.5
U_std = math.sqrt(N_1 * N_2 * (N_1 + N_2 + 1) / 12)


Z = (U - U_mean) / U_std

def standard_normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# two-tailed
alpha = (1 - CONFIDENCE_LEVEL) / 2
p_value = standard_normal_cdf(Z)


if VERBOSE:
    print(f"Expected Mean (mu): {U_mean:.3f}")
    print(f"Standard Deviation (sigma): {U_std:.3f}")
    print(f"Z-score: {Z:.3f}")

print(f"p-value: {p_value:.3f}", end = ' ')

if p_value > alpha:
    print(f"> {alpha:.3f}")
    null_hypothesis = True
else:
    print(f"<= {alpha:.3f}")
    null_hypothesis = False

# Conclusion

print("Conclusion:")
if null_hypothesis:
    print("X1 and X2 have no significant difference in distribution.")
else:
    print("X1 and X2 have different distributions.")

######################################################################
# Compare implementation with an existing implementation

res = stats.mannwhitneyu(X1, X2, alternative='two-sided', method='asymptotic')
print()
print("SciPy Ground Truth:")
print(f"SciPy U-statistic : {res.statistic}")
print(f"SciPy p-value     : {res.pvalue:.3f}")