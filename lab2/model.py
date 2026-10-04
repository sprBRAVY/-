import numpy as np


def normalize_matrix(matrix, criteria_types):
    norm_matrix = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if criteria_types[j] == 1:
            norm_matrix[:, j] = col / np.max(col)
        else:
            norm_matrix[:, j] = np.min(col) / col
    return norm_matrix


def find_pareto_set(matrix):
    n = matrix.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if np.all(matrix[j] >= matrix[i]) and np.any(matrix[j] > matrix[i]):
                    is_pareto[i] = False
                    break
    return is_pareto


def calculate_weights(expert_ranks):
    sums = np.sum(expert_ranks, axis=0)
    return sums / np.sum(sums)


def run_electre(matrix, weights, c_star, d_star):
    n = matrix.shape[0]
    c_matrix = np.zeros((n, n))
    d_matrix = np.zeros((n, n))

    for j in range(n):
        for k in range(n):
            if j != k:
                i_plus = matrix[j] >= matrix[k]
                c_matrix[j, k] = np.sum(weights[i_plus])

                i_minus = matrix[j] < matrix[k]
                if np.any(i_minus):
                    d_matrix[j, k] = np.max(matrix[k][i_minus] - matrix[j][i_minus])
                else:
                    d_matrix[j, k] = 0
            else:
                c_matrix[j, k] = 1.0
                d_matrix[j, k] = 0.0

    c_j = np.zeros(n)
    d_j = np.zeros(n)
    for j in range(n):
        c_j[j] = np.min(c_matrix[j][np.arange(n) != j])
        d_j[j] = np.max(d_matrix[j][np.arange(n) != j])

    core = (c_j > c_star) & (d_j < d_star)
    return c_j, d_j, core