import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA

def normalize_data(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1
    return (x - mean) / std, mean, std

def create_centroids(x, k):
    np.random.seed(9)
    random_indices = np.random.permutation(x.shape[0])
    created_centroids = x[random_indices[:k]]
    return created_centroids

def this_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def clusters_for_points(x, centroids):
    distances = np.zeros((x.shape[0], len(centroids)))
    for i in range(len(centroids)):
        centroid = centroids[i]
        distances[:, i] = np.sqrt(np.sum((x - centroid) ** 2, axis=1))
    return np.argmin(distances, axis=1)

def new_centroids(x, clusters, k):
    centroids = np.zeros((k, x.shape[1]))
    for i in range(k):
        points_in_cluster = x[clusters == i]
        if len(points_in_cluster) > 0:
            centroids[i] = np.mean(points_in_cluster, axis=0)
        else:
            # Если кластер пустой, выбираем случайную точку
            centroids[i] = x[np.random.randint(x.shape[0])]
    return centroids

def has_converged(old_centroids, centroids, tol=1e-4):
    return np.max(np.abs(old_centroids - centroids)) < tol

def plot_clusters(x_normalized, clusters, centroids_normalized):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_normalized)
    centroids_pca = pca.transform(centroids_normalized)

    plt.figure(figsize=(10, 8))

    # Отображаем точки с кластерами
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7, label='Точки данных')

    # Отображаем центроиды
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=300, linewidth=3, label='Центроиды')

    plt.title('K-Means')
    plt.xlabel(wine.data.features.columns[0])
    plt.ylabel(wine.data.features.columns[1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def this_error(x, clusters, centroids):
    wss = 0
    for i in range(len(centroids)):
        cluster_points = x[clusters == i]
        wss += np.sum((cluster_points - centroids[i]) ** 2)
    return wss

def plot_error(error):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(error) + 1), error, marker='o', linewidth=2, c='red')
    plt.title('График WSS по итерациям')
    plt.xlabel('Итерация')
    plt.ylabel('WSS (внутрикластерная сумма квадратов)')
    plt.grid(True, alpha=0.3)
    plt.show()

def kmeans(x, k, max_iters=100):
    # Нормализуем данные
    x_normalized, mean, std = normalize_data(x)

    centroids = create_centroids(x_normalized, k)
    clusters = clusters_for_points(x_normalized, centroids)
    error = []

    for i in range(max_iters):
        old_centroids = centroids.copy()
        clusters = clusters_for_points(x_normalized, centroids)
        centroids = new_centroids(x_normalized, clusters, k)

        error.append(this_error(x_normalized, clusters, centroids))

        if has_converged(old_centroids, centroids):
            print(f"Сходимость достигнута на итерации {i + 1}")
            break

    return clusters, centroids, error, x_normalized

def match_clusters_to_true_labels(clusters, true_labels):
    # Создаем матрицу сопряженности
    contingency_matrix = np.zeros((3, 3), dtype=int)
    for cluster, true_label in zip(clusters, true_labels):
        contingency_matrix[cluster, true_label] += 1

    # Сопоставляем каждый кластер с тем классом, который в нем преобладает
    cluster_to_class = {}
    for cluster in range(3):
        best_class = np.argmax(contingency_matrix[cluster])
        cluster_to_class[cluster] = best_class

    # Переименовываем кластеры согласно сопоставлению
    matched_clusters = np.array([cluster_to_class[cluster] for cluster in clusters])

    return matched_clusters

# Основная программа
wine = fetch_ucirepo(id=109)

X = wine.data.features.values
y = wine.data.targets.values

# Преобразуем метки классов в 0,1,2
clusters_true = []
for i in y:
    if i == 1:
        clusters_true.append(0)
    if i == 2:
        clusters_true.append(1)
    if i == 3:
        clusters_true.append(2)
clusters_true = np.array(clusters_true)

k = 3

clusters, centroids, error, X_normalized = kmeans(X, k)

# Сопоставляем кластеры с реальными классами
clusters_matched = match_clusters_to_true_labels(clusters, clusters_true)

plot_clusters(X_normalized, clusters_matched, centroids)
plot_error(error)

print("Истинные метки:", clusters_true[:178])
print("Предсказанные:", clusters_matched[:178])

# Посчитаем количество в изначальной выборке по классам
i_1 = np.sum(clusters_true == 0)
i_2 = np.sum(clusters_true == 1)
i_3 = np.sum(clusters_true == 2)

# Количество правильно и неправильно кластеризованных
def check_klast(cl):
    right = 0
    wrong = 0
    for i, j in zip(clusters_true, clusters_matched):
        if i == cl and j == cl:
            right += 1
        if i == cl and j != cl:
            wrong += 1
    return right, wrong

right_1, wrong_1 = check_klast(0)
right_2, wrong_2 = check_klast(1)
right_3, wrong_3 = check_klast(2)

# Таблица
data = {
    'Название класса': ['1', '2', '3'],
    'Кол-во точек': [i_1, i_2, i_3],
    'Правильно кластер.': [right_1, right_2, right_3],
    'Неправильно кластер.': [wrong_1, wrong_2, wrong_3]
}

df = pd.DataFrame(data)
print(df)

# Общая точность
total_correct = right_1 + right_2 + right_3
total_samples = len(clusters_true)
accuracy = total_correct / total_samples * 100
print(f"\nОбщая точность: {accuracy:.1f}%")

# import numpy as np
# import pandas as pd
# from ucimlrepo import fetch_ucirepo
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
#
# # Загрузка данных
# wine = fetch_ucirepo(id=109)
# X = wine.data.features.values
# y = wine.data.targets.values
#
# # Преобразуем метки классов в 0,1,2
# clusters_true = []
# for i in y:
#     if i == 1:
#         clusters_true.append(0)
#     elif i == 2:
#         clusters_true.append(1)
#     elif i == 3:
#         clusters_true.append(2)
# clusters_true = np.array(clusters_true)
#
# # Нормализация данных
# scaler = StandardScaler()
# X_normalized = scaler.fit_transform(X)
#
# # K-means с использованием встроенного метода
# kmeans = KMeans(n_clusters=3, random_state=9, n_init=10)
# clusters = kmeans.fit_predict(X_normalized)
# centroids = kmeans.cluster_centers_
#
# # Функция для сопоставления кластеров с реальными классами
# def match_clusters_to_true_labels(clusters, true_labels):
#     contingency_matrix = np.zeros((3, 3), dtype=int)
#     for cluster, true_label in zip(clusters, true_labels):
#         contingency_matrix[cluster, true_label] += 1
#
#     cluster_to_class = {}
#     for cluster in range(3):
#         best_class = np.argmax(contingency_matrix[cluster])
#         cluster_to_class[cluster] = best_class
#
#     matched_clusters = np.array([cluster_to_class[cluster] for cluster in clusters])
#     return matched_clusters
#
# # Сопоставляем кластеры с реальными классами
# clusters_matched = match_clusters_to_true_labels(clusters, clusters_true)
#
# # Вывод результатов
# print("Истинные метки:", clusters_true[:178])
# print("Предсказанные:  ", clusters_matched[:178])
#
# # Статистика по кластерам
# i_1 = np.sum(clusters_true == 0)
# i_2 = np.sum(clusters_true == 1)
# i_3 = np.sum(clusters_true == 2)
#
# def check_klast(cl):
#     right = 0
#     wrong = 0
#     for i, j in zip(clusters_true, clusters_matched):
#         if i == cl and j == cl:
#             right += 1
#         if i == cl and j != cl:
#             wrong += 1
#     return right, wrong
#
# right_1, wrong_1 = check_klast(0)
# right_2, wrong_2 = check_klast(1)
# right_3, wrong_3 = check_klast(2)
#
# # Таблица результатов
# data = {
#     'Название класса': ['1', '2', '3'],
#     'Кол-во точек': [i_1, i_2, i_3],
#     'Правильно кластер.': [right_1, right_2, right_3],
#     'Неправильно кластер.': [wrong_1, wrong_2, wrong_3]
# }
#
# df = pd.DataFrame(data)
# print("\nРезультаты кластеризации:")
# print(df)
#
# # Общая точность
# total_correct = right_1 + right_2 + right_3
# total_samples = len(clusters_true)
# accuracy = total_correct / total_samples * 100
# print(f"\nОбщая точность: {accuracy:.1f}%")