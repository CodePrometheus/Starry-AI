"""
机器学习基础 (10) - PCA 与降维直觉
====================================
这一章解决的问题是：

1. 为什么有些特征虽然有两个，但本质上只表达了一个方向的信息？
2. PCA 为什么要先中心化，再找主成分？
3. 2 维数据投影到 1 维以后，信息到底损失了什么？

本脚本重点演示：
- 降维（dimensionality reduction）的直觉
- 手写 PCA 的关键步骤：中心化、协方差、特征分解
- 把 2 维样本投影到 1 维主成分
- 用一个具体样本走完“中心化 -> 投影 -> 重建”的过程
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 1. 准备一个二维数据集
# ============================================================
# 假设我们在分析 8 位内容创作者的周活跃度。
# 每个样本有两个特征：
#   - short_videos_per_week: 每周发布短视频数量
#   - live_streams_per_week: 每周直播场次
#
# 这两个特征明显相关：
# 发短视频越勤的人，通常直播也更勤。
# 如果它们高度相关，那么 2 维信息里可能有一部分是“重复的”。

print("=" * 60)
print("1. 原始二维数据")
print("=" * 60)

creator_names = [
    "创作者A",
    "创作者B",
    "创作者C",
    "创作者D",
    "创作者E",
    "创作者F",
    "创作者G",
    "创作者H",
]

X = np.array([
    [2.0, 1.0],
    [3.0, 2.0],
    [4.0, 2.0],
    [5.0, 3.0],
    [6.0, 4.0],
    [7.0, 4.0],
    [8.0, 5.0],
    [9.0, 6.0],
], dtype=float)

for name, point in zip(creator_names, X):
    print(
        f"{name}: 每周短视频 {point[0]:>3.1f} 条, "
        f"每周直播 {point[1]:>3.1f} 场"
    )

print("\n直觉观察：这些点大致沿一条斜线分布。")
print("这意味着两个特征虽然是 2 维，但大部分变化可能都集中在同一个主方向上。")


# ============================================================
# 2. PCA 第一步：中心化
# ============================================================
# PCA 默认关心“围绕均值的波动方向”。
# 所以第一步不是直接分解原始数据，而是先减去每一列的均值。
#
# 例如：
#   平均短视频数 = 5.5
#   平均直播数   = 3.375
#
# 那么创作者A的 [2.0, 1.0] 会变成：
#   [2.0 - 5.5, 1.0 - 3.375] = [-3.5, -2.375]

print("\n" + "=" * 60)
print("2. 中心化")
print("=" * 60)

mean = X.mean(axis=0)
X_centered = X - mean

print(f"每一列的均值 = {np.round(mean, 3)}")
print(f"中心化后的数据:\n{np.round(X_centered, 3)}")

first_sample_before = X[0]
first_sample_after = X_centered[0]
print(
    f"\n以 {creator_names[0]} 为例："
    f"\n原始坐标 = {first_sample_before}"
    f"\n中心化后 = {np.round(first_sample_after, 3)}"
)


# ============================================================
# 3. PCA 第二步：看两个特征如何一起变化
# ============================================================
# 协方差矩阵衡量“各个特征是如何共同波动的”。
#
# 对二维数据来说，协方差矩阵长这样：
#   [[var(x1), cov(x1, x2)],
#    [cov(x2, x1), var(x2)]]
#
# 如果 cov 很大且为正，说明两个特征常常同涨同跌。

print("\n" + "=" * 60)
print("3. 协方差矩阵")
print("=" * 60)

cov_matrix = np.cov(X_centered, rowvar=False)
print(f"协方差矩阵:\n{np.round(cov_matrix, 4)}")
print("\n协方差矩阵右上角和左下角都明显大于 0，")
print("说明短视频数和直播场次确实在一起上涨。")


# ============================================================
# 4. PCA 第三步：找方差最大的方向
# ============================================================
# PCA 的目标，是找一个方向 v，让数据投影到这个方向之后，
# 保留的方差信息尽可能多。
#
# 这个方向就是协方差矩阵最大特征值对应的特征向量，
# 也叫第一主成分（principal component 1, PC1）。

print("\n" + "=" * 60)
print("4. 第一主成分")
print("=" * 60)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

principal_component = eigenvectors[:, 0]

# 特征向量的正负号本身没有物理意义。
# 这里统一翻成“向右上方”的方向，便于解释。
if principal_component[0] < 0:
    principal_component = -principal_component

explained_variance_ratio = eigenvalues / eigenvalues.sum()

print(f"特征值（从大到小）= {np.round(eigenvalues, 6)}")
print(f"第一主成分方向 PC1 = {np.round(principal_component, 6)}")
print(f"解释方差比 = {np.round(explained_variance_ratio, 6)}")
print(
    f"\n第一主成分大约解释了 {explained_variance_ratio[0] * 100:.2f}% 的方差信息。"
)
print("这意味着：用 1 个方向描述这批创作者的活跃度，已经能保留绝大多数信息。")


# ============================================================
# 5. 用一个具体样本走完投影过程
# ============================================================
# 投影公式：
#   score = centered_point · principal_component
#
# 这个 score 是一个标量，可以理解成：
#   “这个样本沿着主成分方向走了多远”
#
# 如果再做重建：
#   reconstructed = mean + score * principal_component
#
# 它会回到二维空间，但只能落在主成分这条线上，
# 因此会丢掉垂直于主成分方向的那部分信息。

print("\n" + "=" * 60)
print("5. 手工演示一个样本的投影")
print("=" * 60)

sample_index = 7  # 创作者H
sample_centered = X_centered[sample_index]
sample_score = float(sample_centered @ principal_component)
sample_reconstructed = mean + sample_score * principal_component

print(f"选择样本: {creator_names[sample_index]}")
print(f"原始坐标 = {X[sample_index]}")
print(f"中心化后 = {np.round(sample_centered, 6)}")
print(f"投影分数 score = centered_point · PC1 = {sample_score:.6f}")
print(f"按 PC1 重建回二维 = {np.round(sample_reconstructed, 6)}")
print(
    "\n直觉上，重建点不一定和原始点完全一样，"
    "因为我们只保留了“最重要的一个方向”，"
    "丢掉了垂直方向上的细节。"
)


# ============================================================
# 6. 把所有样本从 2 维压到 1 维
# ============================================================

print("\n" + "=" * 60)
print("6. 所有样本的 1 维表示")
print("=" * 60)


def project(points_centered: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """把中心化后的二维点投影到指定方向。"""
    return points_centered @ direction


def reconstruct(scores: np.ndarray, direction: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """把 1 维分数重建回二维近似点。"""
    return mean + np.outer(scores, direction)


scores_1d = project(X_centered, principal_component)
reconstructed_points = reconstruct(scores_1d, principal_component, mean)

for name, original, score, restored in zip(creator_names, X, scores_1d, reconstructed_points):
    print(
        f"{name}: 原始点 {original} -> 1维分数 {score:>8.4f} "
        f"-> 重建近似点 {np.round(restored, 4)}"
    )

print("\n如果后续模型只需要一个“整体活跃度”指标，")
print("那么这些 1 维分数就可以作为压缩后的新特征。")


# ============================================================
# 7. TODO(human) 练习
# ============================================================

print("\n" + "=" * 60)
print("7. TODO(human) 练习")
print("=" * 60)
print("TODO(human): 把 sample_index 改成 0 或 3，手动核对它们的投影分数。")
print("TODO(human): 试着改用第二主成分 eigenvectors[:, 1] 做投影，观察重建误差为什么会更大。")
