# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from ucimlrepo import fetch_ucirepo
# from sklearn.decomposition import PCA
#
# def create_centroids(X, k):
#     np.random.seed(13)
#     random_indices = np.random.permutation(X.shape[0])
#     centroids = X[random_indices[:k]]
#     return centroids
#
# def this_distance(a, b):
#     return np.sqrt(np.sum((a - b) ** 2))
#
# def clusters_for_points(X, centroids):
#     clusters = []
#     for x in X:
#         distances = [this_distance(x, centroid) for centroid in centroids]
#         cluster = np.argmin(distances)
#         clusters.append(cluster)
#     return np.array(clusters)
#
# def new_centroids(X, clusters, k):
#     centroids = np.zeros((k, X.shape[1]))
#     for i in range(k):
#         points_in_cluster = X[clusters == i]
#         if len(points_in_cluster) > 0:
#             centroids[i] = np.mean(points_in_cluster, axis=0)
#         else:
#             # Если кластер пустой, обновляем центроид случайным образом из оставшихся точек
#             other_points = X[clusters != i]
#             if len(other_points) > 0:
#                 centroids[i] = other_points[np.random.choice(len(other_points)), :]
#     return centroids
#
#
# # Проверка сходимости центроидов
# def has_converged(old_centroids, centroids):
#     return np.allclose(old_centroids, centroids)
#
#
# # Визуализация кластеров
# def plot_clusters(X, clusters, centroids):
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)
#     centroids_pca = pca.transform(centroids)
#
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, label='Кластеры')
#     plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Центроиды')
#
#     plt.title('K-Means алгоритм')
#     plt.xlabel('Компонент 1')
#     plt.ylabel('Компонент 2')
#     plt.legend()
#     plt.show()
#
#
# # Внутрикластерная сумма квадратов (WSS)
# def this_error(X, clusters, centroids):
#     wss = 0
#     for i in range(len(centroids)):
#         cluster_points = X[clusters == i]
#         wss += np.sum((cluster_points - centroids[i]) ** 2)
#     return wss
#
#
# def plot_error(wss_list):
#     plt.figure(figsize=(8, 6))
#     plt.plot(range(1, len(wss_list) + 1), wss_list, marker='o')
#     plt.title('График уменьшения WSS на итерациях')
#     plt.xlabel('Итерация')
#     plt.ylabel('WSS (внутрикластерная сумма квадратов)')
#     plt.show()
#
# def kmeans(X, k, max_iters=5):
#     centroids = create_centroids(X, k)
#     clusters = clusters_for_points(X, centroids)
#     error = []
#
#     # plot_clusters(X, clusters, centroids)
#
#     for i in range(max_iters):
#         old_centroids = centroids
#         clusters = clusters_for_points(X, centroids)
#         centroids = new_centroids(X, clusters, k)
#
#         error.append(this_error(X, clusters, centroids))
#
#         # plot_clusters(X, clusters, centroids)
#         # Проверка сходимости
#         if has_converged(old_centroids, centroids):
#             break
#
#     return clusters, centroids, error
#
#
# wine = fetch_ucirepo(id=109)
#
# X = wine.data.features.values
# y = wine.data.targets.values
#
# k = 3
# clusters, centroids, error = kmeans(X, k)
#
# plot_clusters(X, clusters, centroids)
#
# plot_error(error)
#
# clusters_true = []
# for i in y:
#     if i == 1:
#         clusters_true.append(0)
#     if i == 2:
#         clusters_true.append(1)
#     if i == 3:
#         clusters_true.append(2)
#
# print(clusters_true)
# print(clusters)
#
# # Посчитаем количество в изначально выборке по классам
# i_ve = 0
# i_se = 0
# i_vi = 0
# for i in clusters_true:
#     if i == 0:
#         i_ve += 1
#     if i == 1:
#         i_se += 1
#     if i == 2:
#         i_vi += 1
#
#
# # Посчитаем количество правильно и не правильно кластиризованных
# def check_klast(cl):
#     vo = 0
#     no = 0
#     for i, j in zip(clusters_true, clusters):
#         if i == cl and j == cl:
#             vo += 1
#         if (i == cl and j != cl):
#             no += 1
#     return vo, no
#
#
# vo_se, no_se = check_klast(1)
# vo_ve, no_ve = check_klast(2)
# vo_vi, no_vi = check_klast(0)
#
# # Рисуем табличку
# data = {
#     'Название класса': ['1', '2', '3'],
#     'Кол-во в выб.': [i_ve, i_se, i_vi],
#     'Правильно кластир.': [vo_ve, vo_se, vo_vi],
#     'Не правильно кластир.': [no_ve, no_se, no_vi]
# }
#
# df = pd.DataFrame(data)
#
# print(df)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from ucimlrepo import fetch_ucirepo
# from sklearn.decomposition import PCA
#
#
# def create_centroids(X, k):
#     np.random.seed(9)
#     random_indices = np.random.permutation(X.shape[0])
#     centroids = X[random_indices[:k]]
#     return centroids
#
#
# def this_distance(a, b):
#     return np.sqrt(np.sum((a - b) ** 2))
#
#
# def clusters_for_points(X, centroids):
#     clusters = []
#     for x in X:
#         distances = [this_distance(x, centroid) for centroid in centroids]
#         cluster = np.argmin(distances)
#         clusters.append(cluster)
#     return np.array(clusters)
#
#
# def new_centroids(X, clusters, k):
#     centroids = np.zeros((k, X.shape[1]))
#     for i in range(k):
#         points_in_cluster = X[clusters == i]
#         if len(points_in_cluster) > 0:
#             centroids[i] = np.mean(points_in_cluster, axis=0)
#         else:
#             # Если кластер пустой, обновляем центроид случайным образом из оставшихся точек
#             other_points = X[clusters != i]
#             if len(other_points) > 0:
#                 centroids[i] = other_points[np.random.choice(len(other_points)), :]
#     return centroids
#
#
# # Проверка сходимости центроидов
# def has_converged(old_centroids, centroids):
#     return np.allclose(old_centroids, centroids)
#
#
# # Визуализация кластеров
# def plot_clusters(X, clusters, centroids):
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)
#     centroids_pca = pca.transform(centroids)
#
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, label='Кластеры')
#     plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Центроиды')
#
#     plt.title('K-Means алгоритм')
#     plt.xlabel('Компонент 1')
#     plt.ylabel('Компонент 2')
#     plt.legend()
#     plt.show()
#
#
# # Внутрикластерная сумма квадратов (WSS)
# def this_error(X, clusters, centroids):
#     wss = 0
#     for i in range(len(centroids)):
#         cluster_points = X[clusters == i]
#         wss += np.sum((cluster_points - centroids[i]) ** 2)
#     return wss
#
#
# def plot_error(wss_list):
#     plt.figure(figsize=(8, 6))
#     plt.plot(range(1, len(wss_list) + 1), wss_list, marker='o')
#     plt.title('График уменьшения WSS на итерациях')
#     plt.xlabel('Итерация')
#     plt.ylabel('WSS (внутрикластерная сумма квадратов)')
#     plt.show()
#
#
# def kmeans(X, k, max_iters=100):
#     centroids = create_centroids(X, k)
#     clusters = clusters_for_points(X, centroids)
#     error = []
#
#     for i in range(max_iters):
#         old_centroids = centroids
#         clusters = clusters_for_points(X, centroids)
#         centroids = new_centroids(X, clusters, k)
#
#         error.append(this_error(X, clusters, centroids))
#
#         if has_converged(old_centroids, centroids):
#             break
#
#     return clusters, centroids, error
#
#
# # Функция для сопоставления кластеров с реальными классами
# def match_clusters_to_true_labels(clusters, true_labels):
#     # Создаем матрицу сопряженности
#     contingency_matrix = np.zeros((3, 3), dtype=int)
#     for cluster, true_label in zip(clusters, true_labels):
#         contingency_matrix[cluster, true_label] += 1
#
#     # Сопоставляем каждый кластер с тем классом, который в нем преобладает
#     cluster_to_class = {}
#     for cluster in range(3):
#         best_class = np.argmax(contingency_matrix[cluster])
#         cluster_to_class[cluster] = best_class
#
#     # Переименовываем кластеры согласно сопоставлению
#     matched_clusters = np.array([cluster_to_class[cluster] for cluster in clusters])
#
#     return matched_clusters
#
#
# wine = fetch_ucirepo(id=109)
#
# X = wine.data.features.values
# y = wine.data.targets.values
#
# # Преобразуем метки классов в 0,1,2
# clusters_true = []
# for i in y:
#     if i == 1:
#         clusters_true.append(0)
#     if i == 2:
#         clusters_true.append(1)
#     if i == 3:
#         clusters_true.append(2)
# clusters_true = np.array(clusters_true)
#
# k = 3
# clusters, centroids, error = kmeans(X, k)
#
# # Сопоставляем кластеры с реальными классами
# clusters_matched = match_clusters_to_true_labels(clusters, clusters_true)
#
# plot_clusters(X, clusters_matched, centroids)
# plot_error(error)
#
# print("Истинные метки:", clusters_true[:20])
# print("Предсказанные:", clusters_matched[:20])
#
# # Посчитаем количество в изначальной выборке по классам
# i_ve = np.sum(clusters_true == 0)
# i_se = np.sum(clusters_true == 1)
# i_vi = np.sum(clusters_true == 2)
#
#
# # Количество правильно и неправильно кластеризованных
# def check_klast(cl):
#     vo = 0
#     no = 0
#     for i, j in zip(clusters_true, clusters_matched):
#         if i == cl and j == cl:
#             vo += 1
#         if i == cl and j != cl:
#             no += 1
#     return vo, no
#
#
# vo_ve, no_ve = check_klast(0)
# vo_se, no_se = check_klast(1)
# vo_vi, no_vi = check_klast(2)
#
# # Таблица
# data = {
#     'Название класса': ['1', '2', '3'],
#     'Кол-во в выб.': [i_ve, i_se, i_vi],
#     'Правильно кластир.': [vo_ve, vo_se, vo_vi],
#     'Не правильно кластир.': [no_ve, no_se, no_vi]
# }
#
# df = pd.DataFrame(data)
# print(df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA


def normalize_data(X):
    """Нормализация данных методом Z-score"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Защита от деления на ноль
    std[std == 0] = 1
    return (X - mean) / std


def create_centroids(X, k):
    np.random.seed(13)
    random_indices = np.random.permutation(X.shape[0])
    centroids = X[random_indices[:k]]
    return centroids


def this_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def clusters_for_points(X, centroids):
    clusters = []
    for x in X:
        distances = [this_distance(x, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)


def new_centroids(X, clusters, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points_in_cluster = X[clusters == i]
        if len(points_in_cluster) > 0:
            centroids[i] = np.mean(points_in_cluster, axis=0)
        else:
            # Если кластер пустой, обновляем центроид случайным образом из оставшихся точек
            other_points = X[clusters != i]
            if len(other_points) > 0:
                centroids[i] = other_points[np.random.choice(len(other_points)), :]
    return centroids


# Проверка сходимости центроидов
def has_converged(old_centroids, centroids):
    return np.allclose(old_centroids, centroids)


# Визуализация кластеров
def plot_clusters(X, clusters, centroids, normalized=False):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(centroids)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, label='Кластеры')
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Центроиды')

    title = 'K-Means алгоритм'
    if normalized:
        title += ' (нормализованные данные)'
    plt.title(title)
    plt.xlabel('Компонент 1')
    plt.ylabel('Компонент 2')
    plt.legend()
    plt.show()


# Внутрикластерная сумма квадратов (WSS)
def this_error(X, clusters, centroids):
    wss = 0
    for i in range(len(centroids)):
        cluster_points = X[clusters == i]
        wss += np.sum((cluster_points - centroids[i]) ** 2)
    return wss


def plot_error(wss_list, normalized=False):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(wss_list) + 1), wss_list, marker='o')
    title = 'График уменьшения WSS на итерациях'
    if normalized:
        title += ' (нормализованные данные)'
    plt.title(title)
    plt.xlabel('Итерация')
    plt.ylabel('WSS (внутрикластерная сумма квадратов)')
    plt.show()


def kmeans(X, k, max_iters=100, normalize=False):
    # Нормализуем данные если нужно
    if normalize:
        X_processed = normalize_data(X)
        print("Данные нормализованы")
    else:
        X_processed = X.copy()
        print("Данные не нормализованы")

    centroids = create_centroids(X_processed, k)
    clusters = clusters_for_points(X_processed, centroids)
    error = []

    for i in range(max_iters):
        old_centroids = centroids.copy()  # Важно использовать копию!
        clusters = clusters_for_points(X_processed, centroids)
        centroids = new_centroids(X_processed, clusters, k)

        error.append(this_error(X_processed, clusters, centroids))

        if has_converged(old_centroids, centroids):
            print(f"Сходимость достигнута на итерации {i + 1}")
            break

    return clusters, centroids, error


# Функция для сопоставления кластеров с реальными классами
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

print("=== БЕЗ НОРМАЛИЗАЦИИ ===")
clusters_raw, centroids_raw, error_raw = kmeans(X, k, normalize=False)
clusters_matched_raw = match_clusters_to_true_labels(clusters_raw, clusters_true)
plot_clusters(X, clusters_matched_raw, centroids_raw, normalized=False)
plot_error(error_raw, normalized=False)

print("\n=== С НОРМАЛИЗАЦИЕЙ ===")
clusters_norm, centroids_norm, error_norm = kmeans(X, k, normalize=True)
clusters_matched_norm = match_clusters_to_true_labels(clusters_norm, clusters_true)
# Для визуализации используем исходные данные, но кластеры от нормализованных
plot_clusters(X, clusters_matched_norm, centroids_norm, normalized=True)
plot_error(error_norm, normalized=True)

# Сравним результаты
print("\nСравнение результатов:")
print("Истинные метки:", clusters_true[:20])


def evaluate_clustering(clusters_matched, clusters_true, title):
    print(f"\n{title}:")
    print("Предсказанные:", clusters_matched[:20])

    # Посчитаем количество в изначальной выборке по классам
    i_ve = np.sum(clusters_true == 0)
    i_se = np.sum(clusters_true == 1)
    i_vi = np.sum(clusters_true == 2)

    # Количество правильно и неправильно кластеризованных
    def check_klast(cl):
        vo = 0
        no = 0
        for i, j in zip(clusters_true, clusters_matched):
            if i == cl and j == cl:
                vo += 1
            if i == cl and j != cl:
                no += 1
        return vo, no

    vo_ve, no_ve = check_klast(0)
    vo_se, no_se = check_klast(1)
    vo_vi, no_vi = check_klast(2)

    # Таблица
    data = {
        'Название класса': ['1', '2', '3'],
        'Кол-во в выб.': [i_ve, i_se, i_vi],
        'Правильно кластир.': [vo_ve, vo_se, vo_vi],
        'Не правильно кластир.': [no_ve, no_se, no_vi],
        'Точность %': [
            round(vo_ve / i_ve * 100, 1) if i_ve > 0 else 0,
            round(vo_se / i_se * 100, 1) if i_se > 0 else 0,
            round(vo_vi / i_vi * 100, 1) if i_vi > 0 else 0
        ]
    }

    df = pd.DataFrame(data)
    print(df)

    # Общая точность
    total_correct = vo_ve + vo_se + vo_vi
    total_samples = len(clusters_true)
    accuracy = total_correct / total_samples * 100
    print(f"Общая точность: {accuracy:.1f}%")

    return accuracy


# Оцениваем оба подхода
accuracy_raw = evaluate_clustering(clusters_matched_raw, clusters_true, "БЕЗ НОРМАЛИЗАЦИИ")
accuracy_norm = evaluate_clustering(clusters_matched_norm, clusters_true, "С НОРМАЛИЗАЦИЕЙ")

print(f"\nУлучшение точности: {accuracy_norm - accuracy_raw:.1f}%")
