import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from monte_carlo import run_simulation

# Настройка темы
ctk.set_appearance_mode("Light")

# Цветовая палитра
BG_COLOR = "#FAFAFA"
FRAME_COLOR = "#FFFFFF"
LIGHT_GREEN = "#A8E6CF"
LIGHT_GREEN_HOVER = "#8FCEB7"
LIGHT_ORANGE = "#FFD8B1"
TEXT_COLOR = "#2C3E50"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Монте-Карло: Оптимизация входного контроля")
        self.geometry("600x800")
        self.configure(fg_color=BG_COLOR)

        self.canvas = None  # Переменная для хранения графика
        self.create_widgets()

    def create_widgets(self):
        # Заголовок
        title_lbl = ctk.CTkLabel(self, text="Параметры моделирования", font=("Helvetica", 20, "bold"),
                                 text_color=TEXT_COLOR)
        title_lbl.pack(pady=(20, 10))

        # Фрейм для ввода данных
        self.input_frame = ctk.CTkFrame(self, fg_color=LIGHT_ORANGE, corner_radius=15)
        self.input_frame.pack(padx=20, pady=10, fill="x")

        # Переменные
        self.entries = {}
        self.add_input_row("Доля брака (A, %):", "17", 0)
        self.add_input_row("Стоимость контроля (B, руб):", "7", 1)
        self.add_input_row("Стоимость подгонки (C, руб):", "65", 2)
        self.add_input_row("Стоимость замены (D, руб):", "85", 3)
        self.add_input_row("Кол-во испытаний (N):", "10000", 4)

        # Кнопка запуска
        self.btn_run = ctk.CTkButton(
            self, text="Выполнить расчет",
            font=("Helvetica", 16, "bold"),
            fg_color=LIGHT_GREEN, hover_color=LIGHT_GREEN_HOVER,
            text_color="#1D2A38", corner_radius=10,
            command=self.calculate
        )
        self.btn_run.pack(pady=10)

        # Итоговый вывод
        self.lbl_result = ctk.CTkLabel(self, text="", font=("Helvetica", 16, "bold"), text_color=TEXT_COLOR)
        self.lbl_result.pack(pady=(0, 10))

        # Вкладки для результатов (Таблица и График)
        self.tabview = ctk.CTkTabview(
            self, fg_color=FRAME_COLOR,
            segmented_button_selected_color=LIGHT_GREEN,
            segmented_button_selected_hover_color=LIGHT_GREEN_HOVER,
            segmented_button_unselected_color=FRAME_COLOR,
            text_color=TEXT_COLOR
        )
        self.tabview.pack(padx=20, pady=5, fill="both", expand=True)

        self.tabview.add("Таблица")
        self.tabview.add("График")

        # Настройка фрейма таблицы
        self.table_frame = ctk.CTkScrollableFrame(self.tabview.tab("Таблица"), fg_color=FRAME_COLOR)
        self.table_frame.pack(fill="both", expand=True)

    def add_input_row(self, label_text, default_val, row):
        lbl = ctk.CTkLabel(self.input_frame, text=label_text, font=("Helvetica", 14), text_color=TEXT_COLOR)
        lbl.grid(row=row, column=0, padx=15, pady=8, sticky="w")

        entry = ctk.CTkEntry(self.input_frame, fg_color=FRAME_COLOR, text_color=TEXT_COLOR, border_width=0, width=120)
        entry.insert(0, default_val)
        entry.grid(row=row, column=1, padx=15, pady=8, sticky="e")

        self.input_frame.columnconfigure(1, weight=1)
        self.entries[label_text] = entry

    def calculate(self):
        # Очистка предыдущей таблицы
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        # Сбор данных
        try:
            a = float(self.entries["Доля брака (A, %):"].get())
            b = float(self.entries["Стоимость контроля (B, руб):"].get())
            c = float(self.entries["Стоимость подгонки (C, руб):"].get())
            d = float(self.entries["Стоимость замены (D, руб):"].get())
            n = int(self.entries["Кол-во испытаний (N):"].get())
        except ValueError:
            self.lbl_result.configure(text="Ошибка ввода! Проверьте числа.", text_color="red")
            return

        self.lbl_result.configure(text="Выполнение расчета...", text_color=TEXT_COLOR)
        self.update()

        # Запуск логики
        results = run_simulation(a, b, c, d, n)

        # Подготовка данных
        x_vals = [r[0] for r in results]
        y_vals = [r[1] for r in results]
        min_cost = min(y_vals)
        optimal_proc = x_vals[y_vals.index(min_cost)]

        # --- 1. Заполнение таблицы ---
        hdr_1 = ctk.CTkLabel(self.table_frame, text="Процент контроля", font=("Helvetica", 14, "bold"),
                             text_color=TEXT_COLOR)
        hdr_1.grid(row=0, column=0, padx=20, pady=5, sticky="w")
        hdr_2 = ctk.CTkLabel(self.table_frame, text="Общие затраты (руб)", font=("Helvetica", 14, "bold"),
                             text_color=TEXT_COLOR)
        hdr_2.grid(row=0, column=1, padx=20, pady=5, sticky="e")
        self.table_frame.columnconfigure(0, weight=1)
        self.table_frame.columnconfigure(1, weight=1)

        for i, (proc, cost) in enumerate(results, start=1):
            lbl_p = ctk.CTkLabel(self.table_frame, text=f"{proc}%", font=("Helvetica", 14), text_color=TEXT_COLOR)
            lbl_p.grid(row=i, column=0, padx=20, pady=2, sticky="w")

            # Выделение оптимальной строки зеленым цветом
            color = "#00AA00" if cost == min_cost else TEXT_COLOR
            lbl_c = ctk.CTkLabel(self.table_frame, text=f"{cost:.2f}", font=("Helvetica", 14), text_color=color)
            lbl_c.grid(row=i, column=1, padx=20, pady=2, sticky="e")

        # --- 2. Построение графика ---
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor(FRAME_COLOR)
        ax.set_facecolor(BG_COLOR)

        # Основная линия
        ax.plot(x_vals, y_vals, marker='o', color=TEXT_COLOR, linestyle='-', label='Затраты')

        # Точка минимума
        ax.plot(optimal_proc, min_cost, marker='o', color='red', markersize=8, label='Минимум затрат')

        # Настройки осей и сетки
        ax.set_title("Зависимость затрат от доли входного контроля")
        ax.set_xlabel("Процент контроля (%)")
        ax.set_ylabel("Общие затраты (руб)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        fig.tight_layout()

        # Встраивание графика во вкладку Tkinter
        self.canvas = FigureCanvasTkAgg(fig, master=self.tabview.tab("График"))
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Вывод итога в главный интерфейс
        self.lbl_result.configure(text=f"Оптимальный контроль: {optimal_proc}%  |  Минимум затрат: {min_cost:.2f} руб.")


if __name__ == "__main__":
    app = App()
    app.mainloop()