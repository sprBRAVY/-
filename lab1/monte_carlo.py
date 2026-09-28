from lfsr import LFSR

def run_simulation(a, b, c, d, n=10000):
    rng = LFSR(seed=123456789)
    results = []
    primes = a / 100.0

    for proc in range(0, 101, 5):
        p = proc / 100.0
        total_cost = 0

        for _ in range(n):
            r1 = rng.random()
            r2 = rng.random()

            # Имитация процесса
            if r1 < p:
                total_cost += b
            if (r1 < p) and (r2 < primes):
                total_cost += c
            if (r1 >= p) and (r2 < primes):
                total_cost += d

        results.append((proc, total_cost))

    return results