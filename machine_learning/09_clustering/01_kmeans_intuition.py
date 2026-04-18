"""
机器学习基础 (9) - 聚类与 KMeans 直觉
====================================
这一章解决的问题是：

1. 没有标签时，模型还能从数据里学到什么？
2. KMeans 为什么是“先分组，再更新中心”的循环？
3. 聚类中心到底代表什么？

本脚本重点演示：
- 什么是聚类（clustering）
- 用 2 维小数据集手写 KMeans
- 欧氏距离、样本分配、中心更新
- 惯性（inertia）为什么能衡量“簇内是否紧凑”
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 1. 准备一个没有标签的小数据集
# ============================================================
# 假设我们在做咖啡外卖运营分析。
# 每个用户只有两个观测特征：
#   - weekly_coffee_orders: 每周咖啡下单次数
#   - weekly_dessert_orders: 每周甜点下单次数
#
# 现在没有“高频用户 / 低频用户”的人工标签，
# 我们想让算法自己从行为模式里把用户分成几组。

print("=" * 60)
print("1. 原始用户行为数据")
print("=" * 60)

user_names = [
    "用户A",
    "用户B",
    "用户C",
    "用户D",
    "用户E",
    "用户F",
    "用户G",
    "用户H",
]

X = np.array([
    [1.0, 1.0],   # 用户A：每周 1 杯咖啡，1 份甜点
    [1.5, 2.0],   # 用户B：下单不多
    [2.0, 1.2],   # 用户C：也偏低频
    [2.3, 1.8],   # 用户D：仍然接近低频群体
    [7.5, 8.0],   # 用户E：高频下单
    [8.0, 8.5],   # 用户F：高频下单
    [9.0, 8.0],   # 用户G：高频下单
    [8.5, 9.0],   # 用户H：高频下单
], dtype=float)

for name, point in zip(user_names, X):
    print(
        f"{name}: 每周咖啡 {point[0]:>3.1f} 次, "
        f"每周甜点 {point[1]:>3.1f} 次"
    )

print("\n直觉观察：")
print("前 4 个用户集中在左下角，后 4 个用户集中在右上角。")
print("KMeans 的目标，就是把这种“天然靠近”的样本自动分成簇。")


# ============================================================
# 2. KMeans 的第一步：看谁离哪个中心更近
# ============================================================
# KMeans 的核心循环只有两步：
#   第一步：把每个样本分配给最近的中心
#   第二步：用该簇所有样本的均值更新中心
#
# 这里我们先手动指定 2 个初始中心：
#   - c0 用用户A的位置 [1.0, 1.0]
#   - c1 用用户E的位置 [7.5, 8.0]
#
# 在真实项目里，初始中心常用随机采样或 k-means++。

print("\n" + "=" * 60)
print("2. 第一步：分配到最近中心")
print("=" * 60)


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """计算两个二维点的欧氏距离。"""
    return float(np.linalg.norm(p1 - p2))


def assign_clusters(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """把每个样本分配给最近的聚类中心。"""
    labels = []
    for point in points:
        distances = np.array([euclidean_distance(point, centroid) for centroid in centroids])
        labels.append(int(np.argmin(distances)))
    return np.array(labels, dtype=int)


initial_centroids = np.array([
    [1.0, 1.0],
    [7.5, 8.0],
], dtype=float)

sample_index = 2  # 用户C
sample_point = X[sample_index]
dist_to_c0 = euclidean_distance(sample_point, initial_centroids[0])
dist_to_c1 = euclidean_distance(sample_point, initial_centroids[1])

print(f"初始中心 c0 = {initial_centroids[0]}")
print(f"初始中心 c1 = {initial_centroids[1]}")
print(f"\n拿 {user_names[sample_index]} = {sample_point} 举例：")
print(f"到 c0 的距离 = {dist_to_c0:.3f}")
print(f"到 c1 的距离 = {dist_to_c1:.3f}")
print("因为更接近 c0，所以用户C 会被分到簇 0。")

first_labels = assign_clusters(X, initial_centroids)
print("\n第一次分配结果：")
for name, point, label in zip(user_names, X, first_labels):
    print(f"{name}: 点 {point} -> 簇 {label}")


# ============================================================
# 3. KMeans 的第二步：更新聚类中心
# ============================================================
# 分好组以后，新的中心不是继续用旧值，
# 而是取“这个簇里所有样本坐标的平均值”。
#
# 例如簇 0 如果有 4 个点：
#   [1.0, 1.0], [1.5, 2.0], [2.0, 1.2], [2.3, 1.8]
# 那么新中心就是这 4 个点的按列平均值。

print("\n" + "=" * 60)
print("3. 第二步：更新中心")
print("=" * 60)


def update_centroids(points: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """根据当前簇分配结果，重新计算每个中心。"""
    new_centroids = []
    for cluster_id in range(k):
        cluster_points = points[labels == cluster_id]
        if len(cluster_points) == 0:
            raise ValueError(f"簇 {cluster_id} 为空，当前示例不处理空簇情况。")
        new_centroids.append(cluster_points.mean(axis=0))
    return np.array(new_centroids, dtype=float)


updated_centroids = update_centroids(X, first_labels, k=2)

cluster_0_points = X[first_labels == 0]
cluster_1_points = X[first_labels == 1]

print(f"簇 0 的样本点:\n{cluster_0_points}")
print(f"簇 0 的新中心 = 均值 = {updated_centroids[0]}")
print(f"\n簇 1 的样本点:\n{cluster_1_points}")
print(f"簇 1 的新中心 = 均值 = {updated_centroids[1]}")


# ============================================================
# 4. 把这两步循环起来：完整跑一个 KMeans
# ============================================================
# 当中心基本不再变化时，KMeans 就收敛了。

print("\n" + "=" * 60)
print("4. 完整运行 KMeans")
print("=" * 60)


def compute_inertia(points: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    计算 KMeans 的惯性：
    所有样本到所属中心的平方距离之和。

    惯性越小，通常表示簇内越紧凑。
    """
    squared_distances = []
    for point, label in zip(points, labels):
        squared_distance = euclidean_distance(point, centroids[label]) ** 2
        squared_distances.append(squared_distance)
    return float(np.sum(squared_distances))


def run_kmeans(points: np.ndarray, initial_centroids: np.ndarray, max_iter: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """手写一个最小可运行版本的 KMeans。"""
    centroids = initial_centroids.copy()

    for iteration in range(1, max_iter + 1):
        labels = assign_clusters(points, centroids)
        inertia = compute_inertia(points, labels, centroids)
        print(f"第 {iteration} 轮:")
        print(f"  当前中心:\n{np.round(centroids, 3)}")
        print(f"  当前标签: {labels}")
        print(f"  当前惯性: {inertia:.4f}")

        new_centroids = update_centroids(points, labels, k=len(centroids))
        print(f"  更新后中心:\n{np.round(new_centroids, 3)}")

        if np.allclose(new_centroids, centroids):
            print("  中心没有再变化，算法收敛。")
            return labels, new_centroids

        centroids = new_centroids
        print()

    return labels, centroids


final_labels, final_centroids = run_kmeans(X, initial_centroids)
final_inertia = compute_inertia(X, final_labels, final_centroids)

print("\n最终结果：")
for cluster_id in range(len(final_centroids)):
    member_indices = np.where(final_labels == cluster_id)[0]
    member_names = [user_names[idx] for idx in member_indices]
    member_points = X[member_indices]
    print(f"簇 {cluster_id}: {member_names}")
    print(f"  样本点:\n{member_points}")
    print(f"  最终中心: {np.round(final_centroids[cluster_id], 3)}")

print(f"\n最终惯性 = {final_inertia:.4f}")


# ============================================================
# 5. 如何理解最终中心？
# ============================================================
# KMeans 的中心不是“真实存在的某个用户”，
# 而是这一组用户的平均画像。
#
# 例如簇 0 的中心大约是 [1.7, 1.5]，
# 它可以理解成：
#   “低频用户平均每周下 1.7 次咖啡单、1.5 次甜点单”
#
# 簇 1 的中心大约是 [8.25, 8.375]，
# 它可以理解成：
#   “高频用户平均每周下 8 次左右咖啡单和甜点单”

print("\n" + "=" * 60)
print("5. 聚类结果的业务解释")
print("=" * 60)

for cluster_id, centroid in enumerate(final_centroids):
    print(
        f"簇 {cluster_id} 的平均画像: "
        f"每周咖啡 {centroid[0]:.3f} 次, 每周甜点 {centroid[1]:.3f} 次"
    )

print("\nKMeans 最适合回答的问题不是“这个用户一定是什么类别”，")
print("而是“这批没有标签的用户，看起来能自然分成几群行为模式”。")


# ============================================================
# 6. TODO(human) 练习
# ============================================================

print("\n" + "=" * 60)
print("6. TODO(human) 练习")
print("=" * 60)
print("TODO(human): 把 initial_centroids 改成用户B和用户G，再运行一次，观察是否收敛到同样结果。")
print("TODO(human): 再往 X 里加一个中间用户 [4.5, 5.0]，思考它会被分到哪一簇。")
