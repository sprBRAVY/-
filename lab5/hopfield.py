import numpy as np


class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        """Обучение сети по правилу Хебба с обнулением диагонали"""
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        p = len(patterns)
        for pattern in patterns:
            x = pattern.reshape(-1, 1)
            self.weights += np.dot(x, x.T)

        # Обнуляем диагональ во избежание самовозбуждения
        np.fill_diagonal(self.weights, 0)
        self.weights /= p

    def energy(self, state):
        """Вычисление энергии сети Хопфилда"""
        return -0.5 * np.dot(state.T, np.dot(self.weights, state))

    def predict(self, noisy_pattern, max_iter=50):
        """Асинхронное/синхронное обновление состояний до сходимости"""
        state = noisy_pattern.copy()
        for _ in range(max_iter):
            prev_state = state.copy()
            # Синхронное обновление
            activation = np.dot(self.weights, state)
            state = np.where(activation >= 0, 1.0, -1.0)

            if np.array_equal(state, prev_state):
                break
        return state


def add_noise(pattern, noise_percent):
    """Внесение зашумления (инверсия процента пикселей)"""
    noisy = pattern.copy()
    n = len(noisy)
    n_flip = int(n * (noise_percent / 100.0))

    flip_indices = np.random.choice(n, n_flip, replace=False)
    for idx in flip_indices:
        noisy[idx] *= -1

    return noisy