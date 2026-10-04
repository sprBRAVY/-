import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from data import ALTERNATIVES, CRITERIA_TYPES, EXPERT_RANKS
from model import normalize_matrix, find_pareto_set, calculate_weights, run_electre

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# Цветовая палитра
BG_COLOR = "#FAFAFA"
FRAME_COLOR = "#FFFFFF"
LIGHT_BLUE = "#E3F2FD"
TEXT_COLOR = "#2C3E50"


class ElectreApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Многокритериальный выбор: Метод ЭЛЕКТРА")
        self.geometry("850x650")
        self.configure(fg_color=BG_COLOR)

        self.raw_matrix = np.array([alt["values"] for alt in ALTERNATIVES])
        self.names = np.array([alt["name"] for alt in ALTERNATIVES])
        self.canvas = None

        self.create_widgets()

    def create_widgets(self):
        # Левая панель (Управление и параметры)
        left_panel = ctk.CTkFrame(self, fg_color=FRAME_COLOR, corner_radius=15, width=250)
        left_panel.pack(side="left", fill="y", padx=20, pady=20)
        left_panel.pack_propagate(False)

        ctk.CTkLabel(left_panel, text="Параметры", font=("Helvetica", 18, "bold"), text_color=TEXT_COLOR).pack(
            pady=(20, 10))

        # Вывод рассчитанных весов
        weights = calculate_weights(EXPERT_RANKS)
        weights_frame = ctk.CTkFrame(left_panel, fg_color=LIGHT_BLUE, corner_radius=10)
        weights_frame.pack(padx=15, pady=10, fill="x")
        ctk.CTkLabel(weights_frame, text="Веса критериев:", font=("Helvetica", 12, "bold"), text_color=TEXT_COLOR).pack(
            pady=(5, 0))
        ctk.CTkLabel(weights_frame,
                     text=f"Металл: {weights[0]:.2f}\nСтоимость: {weights[1]:.2f}\nНадежность: {weights[2]:.2f}",
                     text_color=TEXT_COLOR, justify="left").pack(pady=5)

        # Слайдеры
        self.lbl_c = ctk.CTkLabel(left_panel, text="Порог согласия (c*): 0.50", text_color=TEXT_COLOR)
        self.lbl_c.pack(pady=(15, 0))
        self.slider_c = ctk.CTkSlider(left_panel, from_=0.0, to=1.0, command=self.update_c_label)
        self.slider_c.set(0.5)
        self.slider_c.pack(padx=15, pady=5, fill="x")

        self.lbl_d = ctk.CTkLabel(left_panel, text="Порог несогласия (d*): 0.50", text_color=TEXT_COLOR)
        self.lbl_d.pack(pady=(10, 0))
        self.slider_d = ctk.CTkSlider(left_panel, from_=0.0, to=1.0, command=self.update_d_label)
        self.slider_d.set(0.5)
        self.slider_d.pack(padx=15, pady=5, fill="x")

        btn_calc = ctk.CTkButton(left_panel, text="Рассчитать", command=self.calculate, font=("Helvetica", 14, "bold"))
        btn_calc.pack(pady=30, padx=15, fill="x")

        # Правая панель (Вкладки для таблицы и графика)
        right_panel = ctk.CTkTabview(self, fg_color=FRAME_COLOR, text_color=TEXT_COLOR)
        right_panel.pack(side="right", fill="both", expand=True, padx=(0, 20), pady=20)

        right_panel.add("Таблица результатов")
        right_panel.add("Графический анализ")

        self.table_frame = ctk.CTkScrollableFrame(right_panel.tab("Таблица результатов"), fg_color="transparent")
        self.table_frame.pack(fill="both", expand=True)

        self.graph_frame = ctk.CTkFrame(right_panel.tab("Графический анализ"), fg_color="transparent")
        self.graph_frame.pack(fill="both", expand=True)

        self.calculate()  # Первичный расчет

    def update_c_label(self, val):
        self.lbl_c.configure(text=f"Порог согласия (c*): {val:.2f}")

    def update_d_label(self, val):
        self.lbl_d.configure(text=f"Порог несогласия (d*): {val:.2f}")

    def calculate(self):
        c_star = self.slider_c.get()
        d_star = self.slider_d.get()

        norm_matrix = normalize_matrix(self.raw_matrix, CRITERIA_TYPES)
        pareto_mask = find_pareto_set(norm_matrix)
        weights = calculate_weights(EXPERT_RANKS)
        c_j, d_j, core_mask = run_electre(norm_matrix, weights, c_star, d_star)

        # --- 1. Отрисовка Таблицы ---
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        headers = ["Поставщик", "Парето", "Согласие (c)", "Несогласие (d)", "Статус (Ядро)"]
        for col, text in enumerate(headers):
            ctk.CTkLabel(self.table_frame, text=text, font=("Helvetica", 14, "bold"), text_color=TEXT_COLOR).grid(row=0,
                                                                                                                  column=col,
                                                                                                                  padx=10,
                                                                                                                  pady=10,
                                                                                                                  sticky="w")

        for i, name in enumerate(self.names):
            in_pareto = "Да" if pareto_mask[i] else "Нет"
            in_core = "РЕКОМЕНДОВАН" if (pareto_mask[i] and core_mask[i]) else "-"

            # Цветовая индикация
            row_color = "#D4EDDA" if in_core == "РЕКОМЕНДОВАН" else (TEXT_COLOR if pareto_mask[i] else "#999999")
            font_weight = "bold" if in_core == "РЕКОМЕНДОВАН" else "normal"

            ctk.CTkLabel(self.table_frame, text=name, text_color=row_color, font=("Helvetica", 14, font_weight)).grid(
                row=i + 1, column=0, padx=10, pady=5, sticky="w")
            ctk.CTkLabel(self.table_frame, text=in_pareto, text_color=row_color).grid(row=i + 1, column=1, padx=10,
                                                                                      pady=5, sticky="w")

            # Значения c и d показываем только для Парето-оптимальных
            c_val = f"{c_j[i]:.3f}" if pareto_mask[i] else "-"
            d_val = f"{d_j[i]:.3f}" if pareto_mask[i] else "-"

            ctk.CTkLabel(self.table_frame, text=c_val, text_color=row_color).grid(row=i + 1, column=2, padx=10, pady=5,
                                                                                  sticky="w")
            ctk.CTkLabel(self.table_frame, text=d_val, text_color=row_color).grid(row=i + 1, column=3, padx=10, pady=5,
                                                                                  sticky="w")
            ctk.CTkLabel(self.table_frame, text=in_core, text_color=row_color,
                         font=("Helvetica", 14, font_weight)).grid(row=i + 1, column=4, padx=10, pady=5, sticky="w")

        # --- 2. Отрисовка Графика ---
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        fig.patch.set_facecolor(FRAME_COLOR)
        ax.set_facecolor(BG_COLOR)

        # Рисуем все Парето-оптимальные точки
        c_pareto = c_j[pareto_mask]
        d_pareto = d_j[pareto_mask]
        names_pareto = self.names[pareto_mask]

        ax.scatter(c_pareto, d_pareto, color='blue', s=80, zorder=3)

        # Подписи к точкам
        for i, txt in enumerate(names_pareto):
            ax.annotate(txt, (c_pareto[i], d_pareto[i]), xytext=(5, 5), textcoords='offset points', fontsize=10,
                        weight='bold')

        # Пороговые линии
        ax.axvline(x=c_star, color='green', linestyle='--', label=f'Порог c*={c_star:.2f}')
        ax.axhline(y=d_star, color='red', linestyle='--', label=f'Порог d*={d_star:.2f}')

        # Выделение зеленой зоны (Ядро: c > c* и d < d*)
        ax.axvspan(c_star, 1.1, ymin=0, ymax=d_star / 1.1 if d_star < 1.1 else 1, alpha=0.15, color='green',
                   label='Зона ядра')

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Индекс согласия (c) → Больше лучше")
        ax.set_ylabel("Индекс несогласия (d) → Меньше лучше")
        ax.set_title("Поиск оптимальной альтернативы")
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc="upper right", fontsize=8)

        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = ElectreApp()
    app.mainloop()