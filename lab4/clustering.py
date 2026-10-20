import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def run_kmeans(data, k=2, max_iter=100):
    n, m = data.shape
    # Шаг 2: выбор начальных прототипов (равномерно из выборки)
    indices = np.linspace(0, n - 1, k, dtype=int)
    prototypes = data[indices].copy()

    labels = np.zeros(n, dtype=int)
    iterations = 0
    ops_count = 0

    for s in range(1, max_iter + 1):
        iterations = s
        # Шаг 4: отнесение каждого объекта к ближайшему прототипу
        for i in range(n):
            dists = [euclidean_distance(data[i], prototypes[c]) for c in range(k)]
            ops_count += k * m
            labels[i] = int(np.argmin(dists))

        # Шаг 5: вычисление новых прототипов как средних арифметических
        new_prototypes = np.zeros_like(prototypes)
        for c in range(k):
            cluster_points = data[labels == c]
            if len(cluster_points) > 0:
                new_prototypes[c] = np.mean(cluster_points, axis=0)
            else:
                new_prototypes[c] = prototypes[c]
            ops_count += m

        # Шаг 6: проверка условия сходимости P_k^s == P_k^(s-1)
        if np.allclose(prototypes, new_prototypes, atol=1e-6):
            break
        prototypes = new_prototypes

    return {
        "labels": labels,
        "prototypes": prototypes,
        "k": k,
        "iterations": iterations,
        "ops": ops_count,
        "complexity": f"O(s · K · N · M) = {iterations} · {k} · {n} · {m} ≈ {ops_count} опер."
    }


def run_maximin(data):
    n, m = data.shape
    ops_count = 0

    # Шаг 1: первый объект становится прототипом первого кластера
    proto_indices = [0]
    prototypes = [data[0].copy()]

    # Шаг 2: поиск наиболее удаленного объекта от P1 -> P2
    dists_from_p1 = [euclidean_distance(data[0], data[j]) for j in range(n)]
    ops_count += n * m
    second_idx = int(np.argmax(dists_from_p1))
    proto_indices.append(second_idx)
    prototypes.append(data[second_idx].copy())

    labels = np.zeros(n, dtype=int)
    iterations = 0
    threshold_t = 0.0

    while True:
        iterations += 1
        k = len(prototypes)

        # Шаги 3 и 7: расчет порогового расстояния T
        # T = sum_{i=1..K-1} sum_{j=i+1..K} D(Pi, Pj) / (K * (K - 1))
        pair_dists_sum = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                pair_dists_sum += euclidean_distance(prototypes[i], prototypes[j])
                ops_count += m
        threshold_t = pair_dists_sum / (k * (k - 1))

        # Шаг 4: распределение объектов по ближайшим прототипам
        for i in range(n):
            dists = [euclidean_distance(data[i], prototypes[c]) for c in range(k)]
            ops_count += k * m
            labels[i] = int(np.argmin(dists))

        # Шаги 5 и 6: поиск самых удаленных объектов Y_k в каждом кластере и проверка D(P_k, Y_k) < T
        new_protos_added = False
        for c in range(k):
            cluster_indices = np.where(labels == c)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_dists = [euclidean_distance(prototypes[c], data[idx]) for idx in cluster_indices]
            ops_count += len(cluster_indices) * m
            max_local_pos = int(np.argmax(cluster_dists))
            max_dist = cluster_dists[max_local_pos]
            farthest_obj_idx = int(cluster_indices[max_local_pos])

            if max_dist >= threshold_t and farthest_obj_idx not in proto_indices:
                proto_indices.append(farthest_obj_idx)
                prototypes.append(data[farthest_obj_idx].copy())
                new_protos_added = True

        if not new_protos_added:
            break

    return {
        "labels": labels,
        "prototypes": np.array(prototypes),
        "proto_indices": proto_indices,
        "k": len(prototypes),
        "threshold_t": threshold_t,
        "iterations": iterations,
        "ops": ops_count,
        "complexity": f"O(K² · N · M), итераций: {iterations}, ≈ {ops_count} опер."
    }


def classify_point(norm_point, prototypes):
    dists = [euclidean_distance(norm_point, proto) for proto in prototypes]
    best_cluster = int(np.argmin(dists))
    return best_cluster, dists