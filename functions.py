import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


def read_data():
    iris = load_iris()
    # print(iris)
    X = iris.data
    return X


def normalize(X):
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    similarity_matrix = pairwise_distances(X_normalized, metric='euclidean')
    print('===================== similarity matrix =====================')
    print(similarity_matrix)

    plt.figure()
    sns.heatmap(similarity_matrix, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("Similarity Matrix")
    plt.savefig('similarity_matrix.png')
    return X_normalized


def kmeans(X):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    labels = kmeans.labels_
    print(labels)

    # مرکزهای خوشه‌ها
    centers = kmeans.cluster_centers_

    # نمایش خوشه‌بندی
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', s=200)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('K-means Clustering on Iris Dataset')
    plt.savefig('K-means.png')

    score = silhouette_score(X, labels)
    print('score:', score)


def dbscan(X):
    X = X[:, [0, 1]]  # استفاده از ویژگی‌های Sepal Length و Sepal Width

    _dbscan = DBSCAN(eps=0.11, min_samples=2)
    _dbscan.fit(X)

    labels = _dbscan.labels_
    print(labels)

    # تعداد خوشه‌ها (بدون در نظر گرفتن نمونه‌های پرت)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # نمایش خوشه‌بندی
    plt.figure()
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'DBSCAN Clustering on Iris Dataset (Number of Clusters: {n_clusters})')
    plt.legend()
    # plt.show()
    plt.savefig('DBSCAN.png')

    score = silhouette_score(X, labels)
    print('score:', score)


def setMinPtsEpsilonParameters(X):
    # X = X[:, [0, 1]]  # استفاده از ویژگی‌های Sepal Length و Sepal Width

    # تنظیم پارامترهای MinPts و epsilon
    min_pts_values = range(2, 11)
    epsilon_min = np.min(X)
    epsilon_max = np.max(X)
    epsilon_values = np.linspace(epsilon_min, epsilon_max, num=10)[1:]

    best_score = -1
    best_labels = None
    best_min_pts = None
    best_epsilon = None

    # اجرای الگوریتم DBSCAN با پارامترهای مختلف و انتخاب بهترین پارامترها
    for min_pts in min_pts_values:
        for epsilon in epsilon_values:
            dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
            labels = dbscan.fit_predict(X)

            try:
                # ارزیابی کیفیت خوشه‌بندی
                score = silhouette_score(X, labels)

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_min_pts = min_pts
                    best_epsilon = epsilon
            except:
                pass

    # چاپ بهترین پارامترها و برچسب‌های خوشه‌بندی متناظر
    print("Best MinPts:", best_min_pts)
    print("Best Epsilon:", best_epsilon)
    print("Best Clustering Labels:", best_labels)
