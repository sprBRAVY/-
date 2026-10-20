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
        # Панель управления (слева)
        ctrl_frame = ctk.CTkFrame(self, width=280, corner_radius=15)
        ctrl_frame.pack(side="left", fill="y", padx=20, pady=20)

        ctk.CTkLabel(ctrl_frame, text="Параметры теста", font=("Helvetica", 16, "bold")).pack(pady=(20, 10))

        # Выбор образа
        ctk.CTkLabel(ctrl_frame, text="Выберите эталонный образ:").pack(anchor="w", padx=20)
        self.pattern_var = ctk.StringVar(value=PATTERN_NAMES[0])
        for i, name in enumerate(PATTERN_NAMES):
            btn = ctk.CTkRadioButton(ctrl_frame, text=name, variable=self.pattern_var, value=name,
                                     command=self.update_pattern)
            btn.pack(anchor="w", padx=30, pady=2)

        # Выбор процента зашумления
        ctk.CTkLabel(ctrl_frame, text="Процент зашумления: 20%", font=("Helvetica", 12, "bold")).pack(anchor="w",
                                                                                                      padx=20,
                                                                                                      pady=(15, 0))
        self.slider_noise = ctk.CTkSlider(ctrl_frame, from_=10, to=100, number_of_steps=18, command=self.update_noise)
        self.slider_noise.set(20)
        self.slider_noise.pack(padx=20, pady=5, fill="x")

        self.lbl_noise = ctk.CTkLabel(ctrl_frame, text="Текущий шум: 20%")
        self.lbl_noise.pack(anchor="w", padx=20)

        btn_run = ctk.CTkButton(ctrl_frame, text="Запустить распознавание", font=("Helvetica", 14, "bold"),
                                command=self.run_test)
        btn_run.pack(pady=30, padx=20, fill="x")

        # Правая панель визуализации
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

        # Отрисовка графиков (Эталлон | Зашумленный | Восстановленный)
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