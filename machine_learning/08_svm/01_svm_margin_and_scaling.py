"""
机器学习基础 (8) - SVM 的最大间隔与特征缩放
====================================
这一章解决的问题是：

1. SVM 说的“最大间隔”到底在最大什么？
2. 为什么特征缩放对 SVM 这类间隔模型非常重要？
3. 不实现完整优化器的前提下，怎样用一个可运行 demo 理解线性 SVM？

本脚本重点演示：
- 最大间隔的几何直觉
- feature scaling 对训练稳定性的影响
- 线性 SVM 的 hinge loss + 子梯度下降教学版实现
- 支持向量（support vectors）在分类边界附近的作用
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 1. 准备一个线性可分的二分类数据集
# ============================================================
# 场景：根据“近 30 天消费金额 + 到店天数”，判断用户是否值得进入 VIP 运营名单。
#
# 标签我们用 SVM 常见写法：
#   y = +1 表示 VIP 候选
#   y = -1 表示普通用户
#
# 故意让两个特征的量级差很多：
#   - monthly_spend 的范围是 120 ~ 900
#   - visit_days 的范围是 1 ~ 12
#
# 后面我们会直接看到，这个尺度差异会影响训练。

print("=" * 60)
print("1. 线性可分数据")
print("=" * 60)

samples = [
    {"monthly_spend": 120, "visit_days": 1, "label": -1},
    {"monthly_spend": 180, "visit_days": 2, "label": -1},
    {"monthly_spend": 220, "visit_days": 2, "label": -1},
    {"monthly_spend": 260, "visit_days": 3, "label": -1},
    {"monthly_spend": 320, "visit_days": 4, "label": -1},
    {"monthly_spend": 380, "visit_days": 4, "label": -1},
    {"monthly_spend": 520, "visit_days": 7, "label": 1},
    {"monthly_spend": 580, "visit_days": 8, "label": 1},
    {"monthly_spend": 640, "visit_days": 8, "label": 1},
    {"monthly_spend": 720, "visit_days": 10, "label": 1},
    {"monthly_spend": 820, "visit_days": 11, "label": 1},
    {"monthly_spend": 900, "visit_days": 12, "label": 1},
]

for i, sample in enumerate(samples, start=1):
    print(
        f"样本{i:02d}: monthly_spend = {sample['monthly_spend']:>3d}, "
        f"visit_days = {sample['visit_days']:>2d}, label = {sample['label']:>2d}"
    )

feature_names = ["monthly_spend", "visit_days"]
X_raw = np.array([[sample[name] for name in feature_names] for sample in samples], dtype=float)
y = np.array([sample["label"] for sample in samples], dtype=float)


# ============================================================
# 2. 最大间隔 intuition
# ============================================================
# 线性分类器都可以写成：
#
#   score = w @ x + b
#
# 预测时只看 score 的正负：
#   score >= 0 -> +1
#   score <  0 -> -1
#
# SVM 和普通线性分类器的关键区别在于：
#   它不只想“分对”，还想“分得离边界更远”
#
# 对单条样本，几何间隔可以写成：
#
#   margin_i = y_i * (w @ x_i + b) / ||w||
#
# 如果一个超平面把所有样本都分开了，
# 那么最小 margin 越大，说明边界离两类最近样本越远，也就越稳。

print("\n" + "=" * 60)
print("2. 最大间隔 intuition")
print("=" * 60)


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按列标准化。"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    X_scaled = (X - mean) / std_safe
    return X_scaled, mean, std_safe


def minimum_geometric_margin(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
) -> float | None:
    """
    计算最小几何间隔。

    如果某个超平面没有把样本全部分开，返回 None。
    """
    signed_scores = y * (X @ w + b)
    if np.any(signed_scores <= 0):
        return None
    return float(np.min(signed_scores) / np.linalg.norm(w))


X_scaled, feature_mean, feature_std = standardize(X_raw)

candidate_a_w = np.array([1.0, 1.0], dtype=float)
candidate_a_b = 0.0
candidate_b_w = np.array([2.0, 0.2], dtype=float)
candidate_b_b = 0.0

margin_a = minimum_geometric_margin(X_scaled, y, candidate_a_w, candidate_a_b)
margin_b = minimum_geometric_margin(X_scaled, y, candidate_b_w, candidate_b_b)

print("先在标准化后的空间里，比较两条都能分开的候选边界：")
print(f"  候选 A: w = {candidate_a_w}, b = {candidate_a_b}, 最小几何间隔 = {margin_a:.4f}")
print(f"  候选 B: w = {candidate_b_w}, b = {candidate_b_b}, 最小几何间隔 = {margin_b:.4f}")
print("→ 虽然 A 和 B 都能分对，但 A 的最小间隔更大，所以更稳。")


# ============================================================
# 3. 为什么 feature scaling 很重要
# ============================================================
# 现在用同一套 hinge loss + 子梯度下降逻辑训练两次：
#   - 一次直接用原始特征 X_raw
#   - 一次用标准化后的特征 X_scaled
#
# 我们故意保持相同超参数：
#   learning_rate = 0.01
#   C = 3.0
#
# 如果原始特征量级差很多，梯度常常会被大尺度特征主导，
# 训练就更容易抖动、不稳，甚至很难找到好边界。

print("\n" + "=" * 60)
print("3. feature scaling 的影响")
print("=" * 60)

print(f"原始特征均值 = {np.round(feature_mean, 4)}")
print(f"原始特征标准差 = {np.round(feature_std, 4)}")
print(
    "第1条样本标准化后："
    f" [{samples[0]['monthly_spend']}, {samples[0]['visit_days']}] -> "
    f"{np.round(X_scaled[0], 4)}"
)


def predict_sign(scores: np.ndarray) -> np.ndarray:
    """把分数转成 SVM 的类别标签 -1 / +1。"""
    return np.where(scores >= 0, 1.0, -1.0)


def fit_linear_svm_subgradient(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    n_epochs: int,
    C: float,
    snapshot_epochs: set[int],
) -> tuple[np.ndarray, float, list[tuple[int, float, float]]]:
    """
    教学版线性 SVM。

    目标函数：
      0.5 * ||w||^2 + C * mean(max(0, 1 - y * (w @ x + b)))

    这里用的是最基础的子梯度下降，不是完整工业版优化器。
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0
    history: list[tuple[int, float, float]] = []

    for epoch in range(n_epochs):
        scores = X @ w + b
        margins = y * scores
        active_mask = margins < 1.0

        grad_w = w - C * (X[active_mask].T @ y[active_mask]) / n_samples
        grad_b = -C * float(np.sum(y[active_mask])) / n_samples

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if epoch in snapshot_epochs:
            hinge = np.maximum(0.0, 1.0 - margins)
            objective = 0.5 * float(np.dot(w, w)) + C * float(np.mean(hinge))
            preds = predict_sign(scores)
            accuracy = float(np.mean(preds == y))
            history.append((epoch + 1, objective, accuracy))

    return w, b, history


snapshot_epochs = {0, 1, 2, 4, 9, 19, 49, 99, 199}

raw_w, raw_b, raw_history = fit_linear_svm_subgradient(
    X_raw,
    y,
    learning_rate=0.01,
    n_epochs=200,
    C=3.0,
    snapshot_epochs=snapshot_epochs,
)
scaled_w, scaled_b, scaled_history = fit_linear_svm_subgradient(
    X_scaled,
    y,
    learning_rate=0.01,
    n_epochs=200,
    C=3.0,
    snapshot_epochs=snapshot_epochs,
)

print("不做标准化时：")
for epoch, objective, accuracy in raw_history:
    print(f"  第{epoch:3d}轮: objective = {objective:10.4f}, accuracy = {accuracy:.4f}")

print("\n做标准化后：")
for epoch, objective, accuracy in scaled_history:
    print(f"  第{epoch:3d}轮: objective = {objective:10.4f}, accuracy = {accuracy:.4f}")

print("\n→ 同样的学习率下，原始特征训练会明显更抖；标准化后更稳定。")


# ============================================================
# 4. 训练一个更稳定的教学版线性 SVM
# ============================================================
# 下面继续在标准化空间里训练更久一点，看看最终边界、支持向量和预测。

print("\n" + "=" * 60)
print("4. 教学版线性 SVM")
print("=" * 60)

final_w, final_b, _ = fit_linear_svm_subgradient(
    X_scaled,
    y,
    learning_rate=0.01,
    n_epochs=4000,
    C=3.0,
    snapshot_epochs=set(),
)

final_scores = X_scaled @ final_w + final_b
final_predictions = predict_sign(final_scores)
geometric_margins = y * final_scores / np.linalg.norm(final_w)
support_vector_indices = np.argsort(geometric_margins)[:4]

print(f"最终参数: w = {np.round(final_w, 4)}, b = {final_b:.4f}")
print(f"||w|| = {np.linalg.norm(final_w):.4f}")
print(f"训练集预测 = {final_predictions.astype(int)}")
print(f"训练集准确率 = {np.mean(final_predictions == y):.4f}")

print("\n每条样本到分类边界的几何间隔：")
for i, margin in enumerate(geometric_margins, start=1):
    print(
        f"  样本{i:02d}: score = {final_scores[i - 1]:>7.3f}, "
        f"geometric_margin = {margin:.4f}"
    )

print("\n最靠近边界的样本（支持向量候选）索引：")
print(support_vector_indices)
print("→ 它们最先“碰到”间隔边界，对最终超平面位置影响最大。")


# ============================================================
# 5. 用训练好的模型看新用户
# ============================================================

print("\n" + "=" * 60)
print("5. 新样本预测")
print("=" * 60)

new_users_raw = np.array(
    [
        [450.0, 5.0],
        [500.0, 7.0],
        [700.0, 9.0],
    ],
    dtype=float,
)
new_users_scaled = (new_users_raw - feature_mean) / feature_std
new_scores = new_users_scaled @ final_w + final_b
new_preds = predict_sign(new_scores)

for raw_user, score, pred in zip(new_users_raw, new_scores, new_preds):
    print(
        f"  user = [monthly_spend={raw_user[0]:.0f}, visit_days={raw_user[1]:.0f}] -> "
        f"score = {score:.4f}, pred = {int(pred)}"
    )


# ============================================================
# 6. 动手练习
# ============================================================

print("\n" + "=" * 60)
print("6. 动手练习")
print("=" * 60)
print("TODO(human): 把 visit_days 这一列乘以 100，再重新运行第 3 节，观察不做标准化时训练会不会更难。")
print("TODO(human): 只保留 monthly_spend 一个特征，再训练一次，比较支持向量和最终间隔怎么变。")
