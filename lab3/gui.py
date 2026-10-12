import customtkinter as ctk
import numpy as np
from data import X_TRAIN, Y_TRAIN, TEST_OBJ_1
from models import MinDistanceClassifier, PerceptronClassifier


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Распознавание объектов (ЭС)")
        self.geometry("950x650")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.min_dist_clf = MinDistanceClassifier()
        self.perceptron_clf = PerceptronClassifier()

        # Обучаем модели (ВАЖНО: сохраняем результат перцептрона в self.iterations)
        self.min_dist_clf.fit(X_TRAIN, Y_TRAIN)
        self.iterations = self.perceptron_clf.fit(X_TRAIN, Y_TRAIN)

        self._build_ui()

    def _build_ui(self):
        # Левая панель - Данные
        self.left_frame = ctk.CTkFrame(self, width=400)
        self.left_frame.pack(side="left", fill="y", padx=20, pady=20)

        ctk.CTkLabel(self.left_frame, text="Обучающая выборка (12 шт)", font=("Roboto", 18, "bold")).pack(pady=10)

        self.scroll_frame = ctk.CTkScrollableFrame(self.left_frame, width=350, height=500)
        self.scroll_frame.pack(pady=10, padx=10, fill="both", expand=True)

        for i, (x, y) in enumerate(zip(X_TRAIN, Y_TRAIN)):
            bg_color = "#243642" if y == 1 else "#3b2626"
            card = ctk.CTkFrame(self.scroll_frame, fg_color=bg_color, corner_radius=8)
            card.pack(pady=5, padx=5, fill="x")

            header_frame = ctk.CTkFrame(card, fg_color="transparent")
            header_frame.pack(fill="x", padx=10, pady=(10, 5))

            ctk.CTkLabel(header_frame, text=f"Объект {i + 1}", font=("Roboto", 14, "bold")).pack(side="left")

            class_color = "#5ca5ff" if y == 1 else "#ff7a7a"
            ctk.CTkLabel(header_frame, text=f"Класс {y}", font=("Roboto", 14, "bold"), text_color=class_color).pack(
                side="right")

            features_frame = ctk.CTkFrame(card, fg_color="transparent")
            features_frame.pack(fill="x", padx=10, pady=(0, 10))

            formatted_x = "  ".join([f"{val:.1f}" for val in x])
            ctk.CTkLabel(features_frame, text="X: ", font=("Roboto", 12, "bold"), text_color="gray70").pack(side="left")
            ctk.CTkLabel(features_frame, text=f"[{formatted_x}]", font=("Roboto", 12)).pack(side="left")

        # Правая панель - Управление и результаты
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=(0, 20), pady=20)

        ctk.CTkLabel(self.right_frame, text="Ввод признаков (X1 - X5)", font=("Roboto", 18, "bold")).pack(pady=15)

        self.entries = []
        entry_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        entry_frame.pack(pady=5)

        for i in range(5):
            entry = ctk.CTkEntry(entry_frame, width=70, height=35, justify="center", font=("Roboto", 14))
            entry.insert(0, str(TEST_OBJ_1[i]))
            entry.pack(side="left", padx=5)
            self.entries.append(entry)

        self.calc_btn = ctk.CTkButton(self.right_frame, text="Распознать объект", font=("Roboto", 14, "bold"),
                                      height=40, command=self.recognize)
        self.calc_btn.pack(pady=20)

        self.result_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.result_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        # --- Карточка: Метод минимального расстояния ---
        self.res_card_md = ctk.CTkFrame(self.result_frame, corner_radius=10, fg_color="#2b2b2b")
        self.res_card_md.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(self.res_card_md, text="Метод минимального расстояния", font=("Roboto", 16, "bold")).pack(
            anchor="w", padx=15, pady=(15, 5))

        self.md_stats = ctk.CTkFrame(self.res_card_md, fg_color="transparent")
        self.md_stats.pack(fill="x", padx=15, pady=5)
        self.lbl_md_d1 = ctk.CTkLabel(self.md_stats, text="D1: Ожидание...", font=("Consolas", 13), text_color="gray80")
        self.lbl_md_d1.pack(anchor="w")
        self.lbl_md_d2 = ctk.CTkLabel(self.md_stats, text="D2: Ожидание...", font=("Consolas", 13), text_color="gray80")
        self.lbl_md_d2.pack(anchor="w")

        self.md_badge = ctk.CTkFrame(self.res_card_md, corner_radius=6, fg_color="#1e1e1e")
        self.md_badge.pack(fill="x", padx=15, pady=(10, 15))
        self.lbl_md_res = ctk.CTkLabel(self.md_badge, text="РЕЗУЛЬТАТ НЕИЗВЕСТЕН", font=("Roboto", 14, "bold"),
                                       text_color="gray50")
        self.lbl_md_res.pack(pady=8)

        # --- Карточка: Метод алгоритма восприятия ---
        self.res_card_perc = ctk.CTkFrame(self.result_frame, corner_radius=10, fg_color="#2b2b2b")
        self.res_card_perc.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(self.res_card_perc, text="Метод алгоритма восприятия", font=("Roboto", 16, "bold")).pack(
            anchor="w", padx=15, pady=(15, 5))

        self.perc_stats = ctk.CTkFrame(self.res_card_perc, fg_color="transparent")
        self.perc_stats.pack(fill="x", padx=15, pady=5)
        self.lbl_perc_iter = ctk.CTkLabel(self.perc_stats, text="Итераций обучения: --", font=("Consolas", 13),
                                          text_color="gray80")
        self.lbl_perc_iter.pack(anchor="w")
        self.lbl_perc_w = ctk.CTkLabel(self.perc_stats, text="Вектор весов (W): --", font=("Consolas", 13),
                                       text_color="gray80")
        self.lbl_perc_w.pack(anchor="w")
        self.lbl_perc_d12 = ctk.CTkLabel(self.perc_stats, text="D12: Ожидание...", font=("Consolas", 13),
                                         text_color="gray80")
        self.lbl_perc_d12.pack(anchor="w")

        self.perc_badge = ctk.CTkFrame(self.res_card_perc, corner_radius=6, fg_color="#1e1e1e")
        self.perc_badge.pack(fill="x", padx=15, pady=(10, 15))
        self.lbl_perc_res = ctk.CTkLabel(self.perc_badge, text="РЕЗУЛЬТАТ НЕИЗВЕСТЕН", font=("Roboto", 14, "bold"),
                                         text_color="gray50")
        self.lbl_perc_res.pack(pady=8)

    def recognize(self):
        try:
            x_test = [float(e.get().replace(',', '.')) for e in self.entries]
        except ValueError:
            self.lbl_md_res.configure(text="ОШИБКА: ВВЕДИТЕ ЧИСЛА", text_color="#ff7a7a")
            self.md_badge.configure(fg_color="#3b2626")
            self.lbl_perc_res.configure(text="ОШИБКА: ВВЕДИТЕ ЧИСЛА", text_color="#ff7a7a")
            self.perc_badge.configure(fg_color="#3b2626")
            return

        class_1_bg, class_1_fg = "#193c61", "#7abaff"
        class_2_bg, class_2_fg = "#521c1c", "#ff8585"
        error_bg, error_fg = "#521c1c", "#ff8585"

        # 1. Минимальное расстояние
        try:
            c_md, d1, d2 = self.min_dist_clf.predict(x_test)
            self.lbl_md_d1.configure(text=f"D1  =  {d1:.3f}")
            self.lbl_md_d2.configure(text=f"D2  =  {d2:.3f}")

            if c_md == 1:
                self.md_badge.configure(fg_color=class_1_bg)
                self.lbl_md_res.configure(text="ОТНЕСЕН К КЛАССУ 1", text_color=class_1_fg)
            else:
                self.md_badge.configure(fg_color=class_2_bg)
                self.lbl_md_res.configure(text="ОТНЕСЕН К КЛАССУ 2", text_color=class_2_fg)
        except Exception as e:
            self.lbl_md_res.configure(text=f"СБОЙ: {e}", text_color=error_fg)
            self.md_badge.configure(fg_color=error_bg)

        # 2. Перцептрон
        try:
            c_perc, d12 = self.perceptron_clf.predict(x_test)

            # Безопасное извлечение итераций
            iters = getattr(self, 'iterations', 'Неизвестно')

            # Защита от пустого вектора весов
            if self.perceptron_clf.w is not None:
                w_formatted = ", ".join([f"{w:.2f}" for w in self.perceptron_clf.w])
                w_text = f"Веса: W = [{w_formatted}], W0 = {self.perceptron_clf.w0}"
            else:
                w_text = "Веса не были рассчитаны при обучении!"

            self.lbl_perc_iter.configure(text=f"Итераций обучения: {iters}")
            self.lbl_perc_w.configure(text=w_text)
            self.lbl_perc_d12.configure(text=f"Решающая функция: D12 = {d12:.3f}")

            if c_perc == 1:
                self.perc_badge.configure(fg_color=class_1_bg)
                self.lbl_perc_res.configure(text="ОТНЕСЕН К КЛАССУ 1", text_color=class_1_fg)
            else:
                self.perc_badge.configure(fg_color=class_2_bg)
                self.lbl_perc_res.configure(text="ОТНЕСЕН К КЛАССУ 2", text_color=class_2_fg)
        except Exception as e:
            # Если возникнет скрытая ошибка, она выведется прямо на карточку!
            self.lbl_perc_res.configure(text=f"ОШИБКА КОДА: {e}", text_color=error_fg)
            self.perc_badge.configure(fg_color=error_bg)