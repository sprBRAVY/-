# Исследование сети Хопфилда для распознавания зашумленных образов

**Лабораторная работа №5 — Базы знаний и поддержка принятия решений**

**Цель работы:** Исследование свойств ассоциативной памяти искусственной нейронной сети Хопфилда, программная реализация обучения синаптических весов по правилу Хебба и анализ устойчивости сети к зашумлению входных образов в диапазоне от 10% до 100%.

## Описание задачи и теоретическая база

Сеть Хопфилда представляет собой полносвязную искусственную нейронную сеть с обратной связью, функционирующую как ассоциативная память. Основные этапы работы алгоритма:
1. **Обучение (Правило Хебба):** Веса синаптических связей $W_{ij}$ между нейронами вычисляются на основе обучающих эталонных образов (бинарных матриц со значениями $+1$ и $-1$):
   $$W_{ij} = \frac{1}{P} \sum_{m=1}^{P} x_i^m x_j^m \quad (\text{при } i \neq j, \text{ а } W_{ii} = 0)$$
   где $P$ — количество образов, а $x_i^m$ — значение $i$-го нейрона в $m$-м образе.
2. **Функция энергии:** Сеть стремится минимизировать глобальную функцию энергии $E$:
   $$E = -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} W_{ij} s_i s_j$$
   в процессе итеративного обновления состояний до достижения устойчивого состояния (аттрактора).
3. **Распознавание зашумленного образа:** На вход подается искаженный образ. Сеть асинхронно или синхронно пересчитывает состояния нейронов по правилу активации до полной сходимости и восстановления исходного эталона.

---

## Исходные данные (`data.py`)

Модуль содержит матричные представления трех эталонных буквенных образов размерностью 5x7 (всего 35 нейронов), соответствующих инициалам («Б», «Р», «Л»). Пиксели закодированы в виде бинарных векторов со значениями $+1$ (активный пиксель) и $-1$ (фоновый элемент).

```python
import numpy as np

WIDTH = 5
HEIGHT = 7

def parse_grid(grid_str):
    return np.array([1 if char == '#' else -1 for row in grid_str.strip().split('\n') for char in row.strip()], dtype=float)

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
```

---

## Архитектура программного комплекса

Приложение разработано на языке Python с использованием модульной многофайловой архитектуры, разделяющей данные, математическое ядро и пользовательский интерфейс.

### 1. `hopfield.py` (Бизнес-логика и моделирование)

Ядро математической обработки, содержащее логику синаптического обучения по правилу Хебба, функцию расчета глобальной энергии сети, итерационный процесс восстановления образов, а также генератор искажений (`add_noise`), вносящий заданный процент зашумления (инверсии пикселей).

```python
import numpy as np

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        # Обучение сети по правилу Хебба с обнулением главной диагонали
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        p = len(patterns)
        for pattern in patterns:
            x = pattern.reshape(-1, 1)
            self.weights += np.dot(x, x.T)
        
        np.fill_diagonal(self.weights, 0)
        self.weights /= p

    def energy(self, state):
        # Вычисление глобальной энергии сети Хопфилда
        return -0.5 * np.dot(state.T, np.dot(self.weights, state))

    def predict(self, noisy_pattern, max_iter=50):
        # Синхронное обновление состояний нейронов до сходимости (до аттрактора)
        state = noisy_pattern.copy()
        for _ in range(max_iter):
            prev_state = state.copy()
            activation = np.dot(self.weights, state)
            state = np.where(activation >= 0, 1.0, -1.0)
            
            if np.array_equal(state, prev_state):
                break
        return state

def add_noise(pattern, noise_percent):
    # Внесение зашумления путем инверсии заданного процента пикселей
    noisy = pattern.copy()
    n = len(noisy)
    n_flip = int(n * (noise_percent / 100.0))
    
    flip_indices = np.random.choice(n, n_flip, replace=False)
    for idx in flip_indices:
        noisy[idx] *= -1
        
    return noisy
```

### 2. `app.py` (Графический интерфейс пользователя - GUI и точка входа)

Настольное приложение на базе фреймворка **CustomTkinter** и библиотеки **Matplotlib**. Позволяет интерактивно выбирать эталонный образ, настраивать процент зашумления через слайдер в диапазоне от 10% до 100%, запускать тестирование сети и наглядно визуализировать результаты в виде графической сетки.

**Класс `HopfieldApp`**
Наследуется от `ctk.CTk`, связывает интерфейс с модулями `data.py` и `hopfield.py` и управляет графическим выводом:

```python
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from data import TRAIN_PATTERNS, PATTERN_NAMES, WIDTH, HEIGHT
from hopfield import HopfieldNetwork, add_noise

ctk.set_appearance_mode("Light")

class HopfieldApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Сеть Хопфилда: Распознавание зашумленных образов")
        self.geometry("900x700")

        self.net = HopfieldNetwork(WIDTH * HEIGHT)
        self.net.train(TRAIN_PATTERNS)

        self.selected_pattern_idx = 0
        self.noise_percent = 20
        self.canvas = None

        self.create_widgets()
        self.run_test()

    def create_widgets(self):
        ctrl_frame = ctk.CTkFrame(self, width=280, corner_radius=15)
        ctrl_frame.pack(side="left", fill="y", padx=20, pady=20)

        ctk.CTkLabel(ctrl_frame, text="Параметры теста", font=("Helvetica", 16, "bold")).pack(pady=(20, 10))

        ctk.CTkLabel(ctrl_frame, text="Выберите эталонный образ:").pack(anchor="w", padx=20)
        self.pattern_var = ctk.StringVar(value=PATTERN_NAMES[0])
        for i, name in enumerate(PATTERN_NAMES):
            btn = ctk.CTkRadioButton(ctrl_frame, text=name, variable=self.pattern_var, value=name, command=self.update_pattern)
            btn.pack(anchor="w", padx=30, pady=2)

        ctk.CTkLabel(ctrl_frame, text="Процент зашумления: 20%", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=20, pady=(15, 0))
        self.slider_noise = ctk.CTkSlider(ctrl_frame, from_=10, to=100, number_of_steps=18, command=self.update_noise)
        self.slider_noise.set(20)
        self.slider_noise.pack(padx=20, pady=5, fill="x")

        self.lbl_noise = ctk.CTkLabel(ctrl_frame, text="Текущий шум: 20%")
        self.lbl_noise.pack(anchor="w", padx=20)

        btn_run = ctk.CTkButton(ctrl_frame, text="Запустить распознавание", font=("Helvetica", 14, "bold"), command=self.run_test)
        btn_run.pack(pady=30, padx=20, fill="x")

        self.right_frame = ctk.CTkFrame(self, corner_radius=15)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=(0, 20), pady=20)

        self.result_text = ctk.CTkTextbox(self.right_frame, font=("Consolas", 13), height=150)
        self.result_text.pack(padx=20, pady=20, fill="x")

        self.graph_container = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.graph_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def update_pattern(self):
        name = self.pattern_var.get()
        self.selected_pattern_idx = PATTERN_NAMES.index(name)
        self.run_test()

    def update_noise(self, val):
        self.noise_percent = int(val)
        self.lbl_noise.configure(text=f"Текущий шум: {self.noise_percent}%")
        self.run_test()

    def run_test(self):
        original = TRAIN_PATTERNS[self.selected_pattern_idx]
        noisy = add_noise(original, self.noise_percent)
        recovered = self.net.predict(noisy)

        is_correct = np.array_equal(original, recovered)
        
        info = f"Результат тестирования для [{PATTERN_NAMES[self.selected_pattern_idx]}] при шуме {self.noise_percent}%:\n"
        info += f"• Статус распознавания: {'ВЕРНО (Образ восстановлен)' if is_correct else 'ОШИБКА (Ложный аттрактор)'}\n"
        info += f"• Энергия исходного образа: {self.net.energy(original):.2f}\n"
        info += f"• Энергия восстановленного: {self.net.energy(recovered):.2f}"

        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", info)

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, axes = plt.subplots(1, 3, figsize=(6, 3), dpi=100)
        images = [
            (original, "Эталон"),
            (noisy, f"Шум {self.noise_percent}%"),
            (recovered, "Восстановленный")
        ]

        for ax, (img, title) in zip(axes, images):
            ax.imshow(img.reshape(HEIGHT, WIDTH), cmap='binary', vmin=-1, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = HopfieldApp()
    app.mainloop()
```