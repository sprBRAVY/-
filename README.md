# Методы кластерного анализа в экспертных системах

**Лабораторная работа №4 — Базы знаний и поддержка принятия решений**

**Цель работы:** Изучение методов и алгоритмов автоматического разделения множества объектов на кластеры, программная реализация алгоритмов кластеризации методом K-средних и методом максимина, а также оценка их вычислительной сложности.

## Описание задачи и теоретическая база

Методы кластерного анализа предназначены для разделения множества анализируемых объектов $X_1, X_2, \dots, X_N$, каждый из которых описывается вектором из $M$ признаков $X_j = (X_{1j}, X_{2j}, \dots, X_{Mj})$, на группы (кластеры) со сходными характеристиками. В данной работе реализовано два классических алгоритма кластеризации:

1. **Метод K-средних (K-Means):**
   Разделяет выборку на заранее заданное число кластеров $K$. Итеративно пересчитывает центры тяжести (объекты-прототипы $P_k^s$) каждого кластера как среднее арифметическое входящих в него объектов:
   $$P_k^s = \frac{1}{\vert{}C_k\vert{}} \sum_{X_j \in C_k} X_j$$
   Каждый объект относится к кластеру с ближайшим по евклидову расстоянию прототипом. Процесс продолжается до тех пор, пока координаты прототипов не перестанут изменяться на двух соседних итерациях ($P_k^s = P_k^{s-1}$).

2. **Метод максимина (Maximin):**
   Автоматически определяет необходимое количество кластеров $K$ на основе структуры данных. Первым прототипом $P_1$ выбирается произвольный объект, вторым $P_2$ — наиболее удаленный от него. На каждом шаге объекты распределяются по текущим прототипам, находятся наиболее удаленные объекты $Y_k$ внутри каждого кластера и сравниваются с пороговым расстоянием $T$[cite: 3]:
   $$T = \frac{\sum_{i=1}^{K-1} \sum_{j=i+1}^{K} D(P_i, P_j)}{K(K - 1)}$$
   Если для некоторого объекта выполняется условие $D(P_k, Y_k) \ge T$, он становится центром нового кластера ($K = K + 1$), и процедура распределения повторяется.

---

## Исходные данные (`data.py`)

Данные представляют собой структурированную обучающую выборку радиоэлектронных компонентов, состоящую из **12 эталонных образцов**, разделенных по физическим свойствам на две основные группы (кондиционные и склонные к деградации детали). Каждый объект описывается вектором из **3 числовых признаков**:
1. **Отклонение сопротивления $\Delta R$ (%)**
2. **Температурный коэффициент сопротивления ТКС ($10^{-4} \text{ 1/}^\circ\text{C}$)**
3. **Уровень собственных шумов (мкВ/В)**

Поскольку признаки имеют разную физическую размерность, модуль выполняет их предварительное масштабирование (нормализацию) к единому безразмерному диапазону $[0; 1]$ путем деления на максимальные значения по каждому столбцу:

```python
def normalize_data(matrix, test_point=None):
    # Нахождение максимумов по каждому из 3 признаков
    max_vals = np.max(matrix, axis=0)
    # Приведение обучающей матрицы к диапазону [0, 1]
    norm_matrix = matrix / max_vals
    
    if test_point is not None:
        norm_test = np.array(test_point, dtype=float) / max_vals
        return norm_matrix, norm_test, max_vals
    return norm_matrix, max_vals
```

Также в модуле задан вектор контрольного образца `DEFAULT_TEST_OBJECT`, используемый для проверки принадлежности новой детали к сформированным кластерам.

---

## Архитектура программного комплекса

Приложение разработано на языке Python с использованием модульной архитектуры, разделяющей данные, математическое ядро и графический интерфейс.

### 1. `clustering.py` (Бизнес-логика и моделирование)

Ядро математической обработки, содержащее реализацию обоих алгоритмов кластеризации, функцию классификации новых объектов и подсчет элементарных операций для оценки вычислительной сложности.

**Функция `run_kmeans` (Алгоритм K-средних)**
Отвечает за итеративное разбиение объектов на $K$ кластеров и вычисление координат центроидов.

```python
def run_kmeans(data, k=2, max_iter=100):
    n, m = data.shape
    # Выбор начальных прототипов из выборки
    indices = np.linspace(0, n - 1, k, dtype=int)
    prototypes = data[indices].copy()
    
    labels = np.zeros(n, dtype=int)
    ops_count = 0

    for s in range(1, max_iter + 1):
        # Отнесение каждого объекта к ближайшему прототипу
        for i in range(n):
            dists = [euclidean_distance(data[i], prototypes[c]) for c in range(k)]
            ops_count += k * m
            labels[i] = int(np.argmin(dists))

        # Пересчет новых прототипов как средних арифметических кластера
        new_prototypes = np.zeros_like(prototypes)
        for c in range(k):
            cluster_points = data[labels == c]
            if len(cluster_points) > 0:
                new_prototypes[c] = np.mean(cluster_points, axis=0)
            else:
                new_prototypes[c] = prototypes[c]
            ops_count += m

        # Проверка условия сходимости алгоритма
        if np.allclose(prototypes, new_prototypes, atol=1e-6):
            break
        prototypes = new_prototypes

    return {
        "labels": labels, "prototypes": prototypes, "k": k,
        "iterations": s, "ops": ops_count,
        "complexity": f"O(s · K · N · M) = {s} · {k} · {n} · {m} ≈ {ops_count} опер."
    }
```

**Функция `run_maximin` (Алгоритм максимина)**
Реализует динамический поиск числа кластеров через пороговое расстояние $T$[cite: 3] и выделение наиболее удаленных объектов-прототипов.

```python
def run_maximin(data):
    n, m = data.shape
    ops_count = 0

    # Шаг 1-2: выбор P1 и наиболее удаленного от него объекта P2
    proto_indices = [0]
    prototypes = [data[0].copy()]
    dists_from_p1 = [euclidean_distance(data[0], data[j]) for j in range(n)]
    second_idx = int(np.argmax(dists_from_p1))
    proto_indices.append(second_idx)
    prototypes.append(data[second_idx].copy())

    labels = np.zeros(n, dtype=int)
    iterations = 0

    while True:
        iterations += 1
        k = len(prototypes)

        # Расчет порогового расстояния T между всеми парами прототипов
        pair_dists_sum = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                pair_dists_sum += euclidean_distance(prototypes[i], prototypes[j])
        threshold_t = pair_dists_sum / (k * (k - 1))

        # Распределение объектов по ближайшим прототипам
        for i in range(n):
            dists = [euclidean_distance(data[i], prototypes[c]) for c in range(k)]
            labels[i] = int(np.argmin(dists))

        # Поиск кандидатов на новый прототип (проверка условия D(Pk, Yk) >= T)
        new_protos_added = False
        for c in range(k):
            cluster_indices = np.where(labels == c)[0]
            cluster_dists = [euclidean_distance(prototypes[c], data[idx]) for idx in cluster_indices]
            max_local_pos = int(np.argmax(cluster_dists))
            
            if cluster_dists[max_local_pos] >= threshold_t:
                farthest_idx = int(cluster_indices[max_local_pos])
                if farthest_idx not in proto_indices:
                    proto_indices.append(farthest_idx)
                    prototypes.append(data[farthest_idx].copy())
                    new_protos_added = True

        if not new_protos_added:
            break

    return {
        "labels": labels, "prototypes": np.array(prototypes),
        "proto_indices": proto_indices, "k": len(prototypes),
        "threshold_t": threshold_t, "iterations": iterations
    }
```

### 2. `app.py` (Графический интерфейс пользователя - GUI и точка входа)

Настольное приложение на базе библиотек **CustomTkinter** и **Matplotlib**. Обеспечивает интерактивное управление параметрами кластеризации, ввод характеристик распознаваемого объекта и трехмерную визуализацию результатов.

**Класс `ClusteringApp`**
Наследуется от `ctk.CTk`, связывает интерфейс с модулями `data.py` и `clustering.py` и отрисовывает три аналитические вкладки:
* **3D Графики кластеров:** Отображает два трехмерных графика (`projection='3d'`) в пространстве признаков $(X_1, X_2, X_3)$, где цветом выделены сформированные кластеры, маркерами `X` и `D` — найденные прототипы, а золотой звездой `*` — распознаваемый контрольный объект.
* **Таблица распределения:** Выводит исходные и нормализованные значения признаков для всех 12 объектов, сопоставляя номера кластеров, присвоенные обоими методами.
* **Прототипы и сложность:** Отображает точные координаты центров кластеров (в нормализованном и реальном физическом масштабе), а также сводку по итерациям и вычислительной сложности алгоритмов.

```python
class ClusteringApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Кластерный анализ: K-средних и Максимин")
        self.raw_matrix = get_matrix()
        self.names = get_names()
        
        self.create_widgets()
        self.run_analysis() # Первичный расчет при старте программы

    def run_analysis(self):
        # Считывание 3 признаков контрольного объекта и параметра K со слайдера
        test_point = [float(e.get()) for e in self.entries]
        k_val = int(self.slider_k.get())
        
        # Нормализация данных и запуск обоих алгоритмов кластеризации
        norm_matrix, norm_test, max_vals = normalize_data(self.raw_matrix, test_point)
        km_res = run_kmeans(norm_matrix, k=k_val)
        mm_res = run_maximin(norm_matrix)

        # Классификация нового объекта по ближайшему прототипу
        km_cluster, km_dists = classify_point(norm_test, km_res["prototypes"])
        mm_cluster, mm_dists = classify_point(norm_test, mm_res["prototypes"])

        # Обновление 3D-графиков и таблиц интерфейса
        self.draw_graphs(norm_matrix, norm_test, km_res, mm_res, km_cluster, mm_cluster)
        self.populate_table(norm_matrix, km_res, mm_res)
        self.populate_prototypes(km_res, mm_res, max_vals)

if __name__ == "__main__":
    app = ClusteringApp()
    app.mainloop()
```