import numpy as np


class MinDistanceClassifier:
    def __init__(self):
        self.p1 = None
        self.p2 = None

    def fit(self, X, y):
        """Обучение: нахождение прототипов (средних значений) для каждого класса."""
        class_1_objs = X[y == 1]
        class_2_objs = X[y == 2]

        self.p1 = np.mean(class_1_objs, axis=0)
        self.p2 = np.mean(class_2_objs, axis=0)

    def _calculate_d(self, x, p):
        """Вычисление решающей функции D = 2 * sum(Xj*Pj) - sum(Pj^2)"""
        return 2 * np.dot(x, p) - np.dot(p, p)

    def predict(self, x):
        """Распознавание объекта."""
        x = np.array(x)
        d1 = self._calculate_d(x, self.p1)
        d2 = self._calculate_d(x, self.p2)

        predicted_class = 1 if d1 > d2 else 2
        return predicted_class, d1, d2


class PerceptronClassifier:
    def __init__(self):
        self.w = None
        self.w0 = 0

    def fit(self, X, y):
        """Обучение: алгоритм восприятия с коррекцией весов."""
        num_features = X.shape[1]
        self.w = np.zeros(num_features)
        self.w0 = 0

        E = 0
        total_objects = len(X)
        iterations = 0
        max_iterations = 10000  # Защита от зацикливания
        i = 0

        while E < total_objects and iterations < max_iterations:
            idx = i % total_objects
            x_obj = X[idx]
            target_class = y[idx]

            d12 = np.dot(self.w, x_obj) + self.w0

            if target_class == 1:
                if d12 <= 0:
                    self.w = self.w + x_obj
                    self.w0 += 1
                    E = 0
                else:
                    E += 1
            elif target_class == 2:
                if d12 >= 0:
                    self.w = self.w - x_obj
                    self.w0 -= 1
                    E = 0
                else:
                    E += 1

            i += 1
            iterations += 1

        return iterations

    def predict(self, x):
        """Распознавание с помощью построенной разделяющей функции."""
        x = np.array(x)
        d12 = np.dot(self.w, x) + self.w0
        predicted_class = 1 if d12 > 0 else 2
        return predicted_class, d12