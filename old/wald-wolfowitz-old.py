import numpy as np
import math

from statsmodels.sandbox.stats.runs import runstest_2samp

######################################################################
# Configs

SEED = 42

N_1 = 30
MEAN_1 = 500
STD_1 = 100

N_2 = 30
MEAN_2 = 500
STD_2 = 100

TOLERANCE = 1e-6

VERBOSE = True

CONFIDENCE_LEVEL = 0.95
######################################################################
# Data Generation

rng = np.random.default_rng(SEED)
X1, X2 = [], []
def generate_datapoint():
    choice = rng.integers(0, 3, endpoint=True)
    # arbitrary choice of distributions to mix together
    new_datapoint = 0
    if choice == 0:
        new_datapoint = rng.exponential(scale=1/2)
    elif choice == 1:
        new_datapoint = rng.chisquare(df=2)
    elif choice == 2:
        new_datapoint = rng.normal(loc=2, scale=0.2)
    else:
        new_datapoint = rng.geometric(0.5)
    return new_datapoint

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

for i in range(N_1):
    new_datapoint = generate_datapoint()
    X1.append(new_datapoint)
for i in range(N_2):
    new_datapoint = generate_datapoint()
    X2.append(new_datapoint)

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

# Sorting & Merging
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

# Merge while keeping track of group labels (1 or 2)
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

group_num, idx, remaining, bound = 0, 0, [], 0
if idx1 < N_1:
    group_num, idx, remaining, bound = 1, idx1, X1, N_1
if idx2 < N_2:
    group_num, idx, remaining, bound = 2, idx2, X2, N_2
while idx < bound:
    X_merged.append((group_num, remaining[idx]))
    idx += 1

# Computing Runs (R)
R = 1
current_group = X_merged[0][0]

for i in range(1, len(X_merged)):
    if X_merged[i][0] != current_group:
        R += 1
        current_group = X_merged[i][0]

if VERBOSE:
    print(f"Total Runs (R): {R}")

# Transforming to Z score & Comparing to alpha/2

# Expected Mean and Variance for Wald-Wolfowitz
R_mean = (2 * N_1 * N_2) / (N_1 + N_2) + 1
R_var = (2 * N_1 * N_2 * (2 * N_1 * N_2 - N_1 - N_2)) / (((N_1 + N_2) ** 2) * (N_1 + N_2 - 1))
R_std = math.sqrt(R_var)

Z = (R - R_mean) / R_std

def standard_normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

# Two-tailed test: we look at the extreme of whichever side Z falls on
alpha = (1 - CONFIDENCE_LEVEL) / 2
p_value = standard_normal_cdf(-abs(Z))

if VERBOSE:
    print(f"Expected Mean (mu_R): {R_mean:.3f}")
    print(f"Standard Deviation (sigma_R): {R_std:.3f}")
    print(f"Z-score: {Z:.3f}")

print(f"p-value: {p_value:.3f}", end = ' ')

if p_value > alpha:
    print(f"> {alpha:.3f}")
    null_hypothesis = True
else:
    print(f"<= {alpha:.3f}")
    null_hypothesis = False

if null_hypothesis:
    print("X1 and X2 are thoroughly mixed & likely from the same distribution.")
else:
    print("X1 and X2 exhibit unnatural sequences (clumping/alternating) & have different distributions.")

######################################################################
# Compare implementation with an existing implementation
print()
stat, p = runstest_2samp(X1, X2)
print("Statsmodels Ground Truth:")
print(f"Z-score : {stat:.3f}")
print(f"p-value : {p:.3f}")