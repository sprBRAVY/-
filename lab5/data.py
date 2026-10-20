import numpy as np

# Размерность образов: 5 строк на 7 колонок = 35 нейронов
WIDTH = 5
HEIGHT = 7

# Функция конвертации текстовой сетки в массив +1 / -1
def parse_grid(grid_str):
    return np.array([1 if char == '#' else -1 for row in grid_str.strip().split('\n') for char in row.strip()], dtype=float)

# Эталонные образы (3 буквы)
PATTERN_B = parse_grid("""
#####
#....
#....
#####
#....
#....
#####
""")

PATTERN_R = parse_grid("""
#####
#...#
#...#
#####
#..#.
#...#
#...#
""")

PATTERN_L = parse_grid("""
#....
#....
#....
#....
#....
#....
#####
""")

TRAIN_PATTERNS = [PATTERN_B, PATTERN_R, PATTERN_L]
PATTERN_NAMES = ["Буква Б", "Буква Р", "Буква Л"]