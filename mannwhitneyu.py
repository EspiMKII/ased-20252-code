import math

from configs import configs

from scipy import stats

USE_U_FORMULA = configs["use_u_formula"]
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


    U_mean = N_1 * N_2 * 0.5
    U_std = math.sqrt(N_1 * N_2 * (N_1 + N_2 + 1) / 12)


    Z = (U - U_mean) / U_std

    def standard_normal_cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    # two-tailed
    p = 2*standard_normal_cdf(Z)

    return (U, Z, p)

def lib_impl(X_1, X_2):
    res = stats.mannwhitneyu(X_1, X_2, alternative='two-sided', method='asymptotic')
    return (None, res.statistic, res.pvalue)