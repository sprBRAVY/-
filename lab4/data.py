import numpy as np

FEATURE_NAMES = ["Отклонение ΔR (%)", "ТКС (10⁻⁴/°C)", "Уровень шума (мкВ/В)"]

# Обучающее множество (12 объектов, 3 признака)
RAW_OBJECTS = [
    {"name": "Образец 1",  "features": [1.1, 0.45, 0.18]},
    {"name": "Образец 2",  "features": [1.4, 0.52, 0.22]},
    {"name": "Образец 3",  "features": [0.9, 0.38, 0.15]},
    {"name": "Образец 4",  "features": [1.6, 0.60, 0.25]},
    {"name": "Образец 5",  "features": [1.2, 0.49, 0.20]},
    {"name": "Образец 6",  "features": [1.5, 0.55, 0.24]},
    {"name": "Образец 7",  "features": [4.2, 1.85, 0.78]},
    {"name": "Образец 8",  "features": [4.8, 2.10, 0.88]},
    {"name": "Образец 9",  "features": [3.9, 1.70, 0.72]},
    {"name": "Образец 10", "features": [5.1, 2.30, 0.95]},
    {"name": "Образец 11", "features": [4.5, 1.95, 0.82]},
    {"name": "Образец 12", "features": [5.3, 2.40, 0.91]}
]

# Контрольный образец по умолчанию
DEFAULT_TEST_OBJECT = [2.1, 0.85, 0.35]

def get_matrix():
    return np.array([obj["features"] for obj in RAW_OBJECTS], dtype=float)

def get_names():
    return [obj["name"] for obj in RAW_OBJECTS]

def normalize_data(matrix, test_point=None):
    max_vals = np.max(matrix, axis=0)
    norm_matrix = matrix / max_vals
    if test_point is not None:
        norm_test = np.array(test_point, dtype=float) / max_vals
        return norm_matrix, norm_test, max_vals
    return norm_matrix, max_vals