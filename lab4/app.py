import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from data import FEATURE_NAMES, DEFAULT_TEST_OBJECT, get_matrix, get_names, normalize_data
from clustering import run_kmeans, run_maximin, classify_point

ctk.set_appearance_mode("Light")

BG_COLOR = "#FAFAFA"
FRAME_COLOR = "#FFFFFF"
LIGHT_GREEN = "#A8E6CF"
LIGHT_GREEN_HOVER = "#8FCEB7"
LIGHT_ORANGE = "#FFD8B1"
TEXT_COLOR = "#2C3E50"
CLUSTER_COLORS = ["#2980B9", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12", "#16A085"]


class ClusteringApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Кластерный анализ: K-средних и Максимин")
        self.geometry("1100x720")
        self.configure(fg_color=BG_COLOR)

        self.raw_matrix = get_matrix()
        self.names = get_names()
        self.canvas = None

        self.create_widgets()
        self.run_analysis()

    def create_widgets(self):
        # Левая панель управления
        left_panel = ctk.CTkFrame(self, fg_color=FRAME_COLOR, corner_radius=15, width=320)
        left_panel.pack(side="left", fill="y", padx=15, pady=15)
        left_panel.pack_propagate(False)

        ctk.CTkLabel(left_panel, text="Параметры кластеризации", font=("Helvetica", 17, "bold"),
                     text_color=TEXT_COLOR).pack(pady=(15, 10))

        # Настройка K для K-средних
        k_frame = ctk.CTkFrame(left_panel, fg_color=LIGHT_ORANGE, corner_radius=10)
        k_frame.pack(padx=15, pady=5, fill="x")

        self.lbl_k = ctk.CTkLabel(k_frame, text="Число кластеров K (K-средних): 2",
                                  font=("Helvetica", 13, "bold"), text_color=TEXT_COLOR)
        self.lbl_k.pack(pady=(8, 2))

        self.slider_k = ctk.CTkSlider(k_frame, from_=2, to=5, number_of_steps=3, command=self.update_k_label)
        self.slider_k.set(2)
        self.slider_k.pack(padx=15, pady=(0, 10), fill="x")

        # Ввод нового распознаваемого объекта (3 признака)
        obj_frame = ctk.CTkFrame(left_panel, fg_color=LIGHT_ORANGE, corner_radius=10)
        obj_frame.pack(padx=15, pady=10, fill="x")

        ctk.CTkLabel(obj_frame, text="Распознаваемый объект (3 признака)",
                     font=("Helvetica", 13, "bold"), text_color=TEXT_COLOR).pack(pady=(8, 4))

        self.entries = []
        for idx, fname in enumerate(FEATURE_NAMES):
            row = ctk.CTkFrame(obj_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=3)
            ctk.CTkLabel(row, text=fname, font=("Helvetica", 12), text_color=TEXT_COLOR).pack(side="left")
            ent = ctk.CTkEntry(row, width=75, fg_color=FRAME_COLOR, text_color=TEXT_COLOR, border_width=0)
            ent.insert(0, str(DEFAULT_TEST_OBJECT[idx]))
            ent.pack(side="right")
            self.entries.append(ent)

        btn_run = ctk.CTkButton(
            left_panel, text="Выполнить кластеризацию",
            font=("Helvetica", 14, "bold"),
            fg_color=LIGHT_GREEN, hover_color=LIGHT_GREEN_HOVER,
            text_color="#1D2A38", corner_radius=10,
            command=self.run_analysis
        )
        btn_run.pack(pady=12, padx=15, fill="x")

        # Карточка сводки и оценки сложности
        self.summary_box = ctk.CTkFrame(left_panel, fg_color="#F0F4F8", corner_radius=10)
        self.summary_box.pack(padx=15, pady=5, fill="both", expand=True)

        self.lbl_summary = ctk.CTkLabel(
            self.summary_box, text="", font=("Helvetica", 12),
            text_color=TEXT_COLOR, justify="left", wraplength=270
        )
        self.lbl_summary.pack(padx=10, pady=10, anchor="nw")

        # Правая панель с вкладками
        self.tabview = ctk.CTkTabview(
            self, fg_color=FRAME_COLOR,
            segmented_button_selected_color=LIGHT_GREEN,
            segmented_button_selected_hover_color=LIGHT_GREEN_HOVER,
            segmented_button_unselected_color=BG_COLOR,
            text_color=TEXT_COLOR
        )
        self.tabview.pack(side="right", fill="both", expand=True, padx=(0, 15), pady=15)

        self.tabview.add("3D Графики кластеров")
        self.tabview.add("Таблица распределения")
        self.tabview.add("Прототипы и сложность")

        self.graph_frame = ctk.CTkFrame(self.tabview.tab("3D Графики кластеров"), fg_color="transparent")
        self.graph_frame.pack(fill="both", expand=True)

        self.table_frame = ctk.CTkScrollableFrame(self.tabview.tab("Таблица распределения"), fg_color="transparent")
        self.table_frame.pack(fill="both", expand=True)

        self.proto_frame = ctk.CTkScrollableFrame(self.tabview.tab("Прототипы и сложность"), fg_color="transparent")
        self.proto_frame.pack(fill="both", expand=True)

    def update_k_label(self, val):
        self.lbl_k.configure(text=f"Число кластеров K (K-средних): {int(val)}")

    def run_analysis(self):
        try:
            test_point = [float(e.get()) for e in self.entries]
        except ValueError:
            self.lbl_summary.configure(text="Ошибка: проверьте корректность чисел в полях признаков.", text_color="red")
            return

        k_val = int(self.slider_k.get())
        norm_matrix, norm_test, max_vals = normalize_data(self.raw_matrix, test_point)

        km_res = run_kmeans(norm_matrix, k=k_val)
        mm_res = run_maximin(norm_matrix)

        km_cluster, km_dists = classify_point(norm_test, km_res["prototypes"])
        mm_cluster, mm_dists = classify_point(norm_test, mm_res["prototypes"])

        summary_text = (
            f"РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ:\n"
            f"• K-средних: Кластер #{km_cluster + 1}\n"
            f"  (мин. расст. D = {km_dists[km_cluster]:.3f})\n"
            f"• Максимин: Кластер #{mm_cluster + 1}\n"
            f"  (мин. расст. D = {mm_dists[mm_cluster]:.3f})\n\n"
            f"ПАРАМЕТРЫ АЛГОРИТМОВ:\n"
            f"• K-средних:\n"
            f"  Кластеров K = {km_res['k']}, итераций = {km_res['iterations']}\n"
            f"• Максимин:\n"
            f"  Найдено кластеров K = {mm_res['k']}\n"
            f"  Порог T = {mm_res['threshold_t']:.4f}\n\n"
            f"ВЫЧИСЛИТЕЛЬНАЯ СЛОЖНОСТЬ:\n"
            f"• K-средних: {km_res['complexity']}\n"
            f"• Максимин: {mm_res['complexity']}"
        )
        self.lbl_summary.configure(text=summary_text, text_color=TEXT_COLOR)

        self.draw_graphs(norm_matrix, norm_test, km_res, mm_res, km_cluster, mm_cluster)
        self.populate_table(norm_matrix, km_res, mm_res)
        self.populate_prototypes(km_res, mm_res, max_vals)

    def draw_graphs(self, norm_matrix, norm_test, km_res, mm_res, km_c, mm_c):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig = plt.figure(figsize=(8.5, 4.5), dpi=100)
        fig.patch.set_facecolor(FRAME_COLOR)

        # 1. График K-средних
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_facecolor(BG_COLOR)
        for c in range(km_res["k"]):
            pts = norm_matrix[km_res["labels"] == c]
            color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            if len(pts) > 0:
                ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=45, label=f"Кластер {c+1}")
            proto = km_res["prototypes"][c]
            ax1.scatter(proto[0], proto[1], proto[2], color=color, marker="X", s=130, edgecolors="black")

        ax1.scatter(norm_test[0], norm_test[1], norm_test[2], color="#F1C40F", marker="*", s=220,
                    edgecolors="black", label=f"Объект (Кл.{km_c+1})")
        ax1.set_title(f"Метод K-средних (K={km_res['k']})", fontsize=10, weight="bold")
        ax1.set_xlabel("X1 (ΔR)", fontsize=8)
        ax1.set_ylabel("X2 (ТКС)", fontsize=8)
        ax1.set_zlabel("X3 (Шум)", fontsize=8)
        ax1.legend(loc="upper left", fontsize=7)

        # 2. График Максимина
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_facecolor(BG_COLOR)
        for c in range(mm_res["k"]):
            pts = norm_matrix[mm_res["labels"] == c]
            color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            if len(pts) > 0:
                ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=45, label=f"Кластер {c+1}")
            proto = mm_res["prototypes"][c]
            ax2.scatter(proto[0], proto[1], proto[2], color=color, marker="D", s=110, edgecolors="black")

        ax2.scatter(norm_test[0], norm_test[1], norm_test[2], color="#F1C40F", marker="*", s=220,
                    edgecolors="black", label=f"Объект (Кл.{mm_c+1})")
        ax2.set_title(f"Метод Максимина (Авто K={mm_res['k']}, T={mm_res['threshold_t']:.2f})", fontsize=10, weight="bold")
        ax2.set_xlabel("X1 (ΔR)", fontsize=8)
        ax2.set_ylabel("X2 (ТКС)", fontsize=8)
        ax2.set_zlabel("X3 (Шум)", fontsize=8)
        ax2.legend(loc="upper left", fontsize=7)

        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def populate_table(self, norm_matrix, km_res, mm_res):
        for w in self.table_frame.winfo_children():
            w.destroy()

        headers = ["Объект", "Признаки (исх.)", "Норм. значения", "K-средних", "Максимин"]
        for col, h in enumerate(headers):
            ctk.CTkLabel(self.table_frame, text=h, font=("Helvetica", 13, "bold"),
                         text_color=TEXT_COLOR).grid(row=0, column=col, padx=8, pady=6, sticky="w")

        for i, name in enumerate(self.names):
            raw_str = ", ".join(f"{v:.2f}" for v in self.raw_matrix[i])
            norm_str = ", ".join(f"{v:.2f}" for v in norm_matrix[i])
            km_c = km_res["labels"][i]
            mm_c = mm_res["labels"][i]

            ctk.CTkLabel(self.table_frame, text=name, text_color=TEXT_COLOR).grid(row=i+1, column=0, padx=8, pady=3, sticky="w")
            ctk.CTkLabel(self.table_frame, text=raw_str, text_color=TEXT_COLOR).grid(row=i+1, column=1, padx=8, pady=3, sticky="w")
            ctk.CTkLabel(self.table_frame, text=norm_str, text_color=TEXT_COLOR).grid(row=i+1, column=2, padx=8, pady=3, sticky="w")
            ctk.CTkLabel(self.table_frame, text=f"Кластер #{km_c + 1}",
                         text_color=CLUSTER_COLORS[km_c % len(CLUSTER_COLORS)],
                         font=("Helvetica", 12, "bold")).grid(row=i+1, column=3, padx=8, pady=3, sticky="w")
            ctk.CTkLabel(self.table_frame, text=f"Кластер #{mm_c + 1}",
                         text_color=CLUSTER_COLORS[mm_c % len(CLUSTER_COLORS)],
                         font=("Helvetica", 12, "bold")).grid(row=i+1, column=4, padx=8, pady=3, sticky="w")

    def populate_prototypes(self, km_res, mm_res, max_vals):
        for w in self.proto_frame.winfo_children():
            w.destroy()

        row_idx = 0
        ctk.CTkLabel(self.proto_frame, text="Координаты центров кластеров (Прототипы K-средних)",
                     font=("Helvetica", 14, "bold"), text_color=TEXT_COLOR).grid(row=row_idx, column=0, sticky="w", padx=10, pady=(10, 5))
        row_idx += 1

        for c, proto in enumerate(km_res["prototypes"]):
            real_proto = proto * max_vals
            txt = f"Прототип P{c+1}: норм. [{proto[0]:.3f}, {proto[1]:.3f}, {proto[2]:.3f}]  |  физ. [{real_proto[0]:.2f}, {real_proto[1]:.2f}, {real_proto[2]:.2f}]"
            ctk.CTkLabel(self.proto_frame, text=txt, text_color=TEXT_COLOR).grid(row=row_idx, column=0, sticky="w", padx=15, pady=2)
            row_idx += 1

        ctk.CTkLabel(self.proto_frame, text="Координаты прототипов (Метод Максимина)",
                     font=("Helvetica", 14, "bold"), text_color=TEXT_COLOR).grid(row=row_idx, column=0, sticky="w", padx=10, pady=(15, 5))
        row_idx += 1

        for c, proto in enumerate(mm_res["prototypes"]):
            obj_name = self.names[mm_res["proto_indices"][c]]
            txt = f"Прототип P{c+1} ({obj_name}): норм. [{proto[0]:.3f}, {proto[1]:.3f}, {proto[2]:.3f}]"
            ctk.CTkLabel(self.proto_frame, text=txt, text_color=TEXT_COLOR).grid(row=row_idx, column=0, sticky="w", padx=15, pady=2)
            row_idx += 1


if __name__ == "__main__":
    app = ClusteringApp()
    app.mainloop()