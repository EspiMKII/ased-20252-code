import math

from configs import configs

from statsmodels.sandbox.stats.runs import runstest_2samp

VERBOSE = configs['verbose']

def self_impl(X_1, X_2):
    N_1, N_2 = len(X_1), len(X_2)

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

    X_1 = merge_sort(X_1, N_1, 0, N_1 - 1)
    X_2 = merge_sort(X_2, N_2, 0, N_2 - 1)

    X_merged = []
    idx1, idx2 = 0, 0
    while idx1 < N_1 and idx2 < N_2:
        if X_1[idx1] <= X_2[idx2]:
            temp = (1, X_1[idx1])
            idx1 += 1
        else:
            temp = (2, X_2[idx2])
            idx2 += 1
        X_merged.append(temp)
    group_num ,idx, remaining, bound = 0, 0, [], 0
    if idx1 < N_1:
        group_num, idx, remaining, bound = 1, idx1, X_1, N_1
    if idx2 < N_2:
        group_num, idx, remaining, bound = 2, idx2, X_2, N_2
    while idx < bound:
        X_merged.append((group_num ,remaining[idx]))
        idx += 1

    U_1, U_2 = 0, 0
    ranked_1, ranked_2 = [], []
    for i in range(len(X_merged)):
        num, rank = X_merged[i], i+1
        if num[0] == 1:
            ranked_1.append(rank)
        if num[0] == 2:
            ranked_2.append(rank)

    R = 1
    current_group = X_merged[0][0]

    for i in range(1, len(X_merged)):
        if X_merged[i][0] != current_group:
            R += 1
            current_group = X_merged[i][0]

    R_mean = (2 * N_1 * N_2) / (N_1 + N_2) + 1
    R_var = (2 * N_1 * N_2 * (2 * N_1 * N_2 - N_1 - N_2)) / (((N_1 + N_2) ** 2) * (N_1 + N_2 - 1))
    R_std = math.sqrt(R_var)

    Z = (R - R_mean) / R_std

    def standard_normal_cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    # Two-tailed test: we look at the extreme of whichever side Z falls on
    p = 2*standard_normal_cdf(-abs(Z))

    return (R, Z, p)

def lib_impl(X_1, X_2):
    stat, p = runstest_2samp(X_1, X_2)
    return (None, stat, p)