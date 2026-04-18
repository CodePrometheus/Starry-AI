"""
机器学习基础 (6) - 逻辑回归
====================================
这一章解决的问题是：

1. 什么叫二分类问题？
2. logit 和 sigmoid 为什么能把线性打分变成概率？
3. 逻辑回归如何用交叉熵和梯度下降学出分类边界？

本脚本重点演示：
- 二分类任务与概率输出
- sigmoid 函数与 logit
- 阈值 threshold 对分类结果的影响
- 二分类交叉熵
- 用梯度下降从零训练一个逻辑回归模型
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 1. 准备一个二分类数据集
# ============================================================
# 场景：用 "刷题时长 + 模拟面试分数" 预测候选人是否通过一面。
#
# passed = 1 表示通过
# passed = 0 表示未通过
#
# 这和线性回归的区别是：
#   - 线性回归输出连续值，例如房价 520 万
#   - 逻辑回归输出概率，例如通过概率 0.82

print("=" * 60)
print("1. 二分类数据")
print("=" * 60)

samples = [
    {"study_hours": 1.0, "mock_score": 45, "passed": 0},
    {"study_hours": 1.5, "mock_score": 50, "passed": 0},
    {"study_hours": 2.0, "mock_score": 48, "passed": 0},
    {"study_hours": 2.5, "mock_score": 60, "passed": 0},
    {"study_hours": 3.0, "mock_score": 65, "passed": 0},
    {"study_hours": 3.5, "mock_score": 70, "passed": 1},
    {"study_hours": 4.0, "mock_score": 72, "passed": 1},
    {"study_hours": 4.5, "mock_score": 80, "passed": 1},
    {"study_hours": 5.0, "mock_score": 85, "passed": 1},
    {"study_hours": 5.5, "mock_score": 90, "passed": 1},
]

for i, sample in enumerate(samples, start=1):
    print(
        f"样本{i:02d}: study_hours = {sample['study_hours']:.1f}, "
        f"mock_score = {sample['mock_score']:>2d}, passed = {sample['passed']}"
    )


# 模型训练时，我们把分数缩放到 0~1，避免不同特征量级差太大。
# 例如第 1 条样本的 mock_score = 45，会变成 0.45。
X = np.array(
    [[sample["study_hours"], sample["mock_score"] / 100.0] for sample in samples],
    dtype=float,
)
y = np.array([sample["passed"] for sample in samples], dtype=float)

print("\n训练矩阵 X（第二列已经把 mock_score 除以 100）:")
print(np.round(X, 3))
print(f"标签 y = {y.astype(int)}")


# ============================================================
# 2. sigmoid 和 logit
# ============================================================
# 逻辑回归先做线性打分：
#
#   z = w1 * x1 + w2 * x2 + b
#
# 这个 z 就叫 logit，也可以理解为“还没过 sigmoid 的原始分数”。
# 然后再用 sigmoid 把它压到 0~1 之间：
#
#   sigmoid(z) = 1 / (1 + e^(-z))
#
# 关键直觉：
#   - z = 0    -> 概率 0.5
#   - z > 0    -> 更偏向正类
#   - z < 0    -> 更偏向负类

print("\n" + "=" * 60)
print("2. sigmoid 和 logit")
print("=" * 60)


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """把任意实数压缩到 0~1 之间。"""
    return 1.0 / (1.0 + np.exp(-z))


def probability_to_logit(probability: float) -> float:
    """
    概率转回 logit。

    logit(p) = log(p / (1 - p))
    """
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return np.log(probability / (1 - probability))


guess_w = np.array([1.5, 4.0], dtype=float)
guess_b = -4.0
first_x = X[0]
first_logit = first_x @ guess_w + guess_b
first_probability = sigmoid(first_logit)

print(f"先随便猜一组参数: w = {guess_w}, b = {guess_b}")
print(
    "第1条样本代入："
    f"z = 1.5 * {first_x[0]:.2f} + 4.0 * {first_x[1]:.2f} + ({guess_b})"
    f" = {first_logit:.4f}"
)
print(f"sigmoid({first_logit:.4f}) = {first_probability:.4f}")
print(f"→ 模型认为这位候选人通过一面的概率约是 {first_probability:.2%}")

example_probability = 0.8
example_logit = probability_to_logit(example_probability)
print(f"\n反过来看：如果模型输出概率 p = {example_probability:.1f}")
print(f"logit(p) = log(p / (1 - p)) = {example_logit:.4f}")
print(f"再过一次 sigmoid：sigmoid({example_logit:.4f}) = {sigmoid(example_logit):.4f}")


# ============================================================
# 3. 阈值 threshold：概率怎么变成 0 / 1 分类
# ============================================================
# 逻辑回归本质上先输出概率。
# 真正变成“通过 / 不通过”时，还需要一个阈值。
#
# 最常见的是 threshold = 0.5：
#   p >= 0.5 -> 预测为 1
#   p <  0.5 -> 预测为 0
#
# 但阈值不是永远固定的：
#   - 想少放过风险样本，可以把阈值调高
#   - 想少漏掉正样本，可以把阈值调低

print("\n" + "=" * 60)
print("3. 阈值 threshold")
print("=" * 60)


def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """把概率按阈值转换成 0 / 1。"""
    return (probabilities >= threshold).astype(int)


guess_logits = X @ guess_w + guess_b
guess_probabilities = sigmoid(guess_logits)
pred_at_05 = apply_threshold(guess_probabilities, threshold=0.5)
pred_at_07 = apply_threshold(guess_probabilities, threshold=0.7)

print("用刚才随便猜的参数，对全部样本先算概率：")
for i, (sample, probability) in enumerate(zip(samples, guess_probabilities), start=1):
    print(
        f"  样本{i:02d}: study_hours = {sample['study_hours']:.1f}, "
        f"mock_score = {sample['mock_score']:>2d}, p(passed=1) = {probability:.4f}"
    )

print("\nthreshold = 0.5 时的分类结果:")
print(pred_at_05)
print("threshold = 0.7 时的分类结果:")
print(pred_at_07)
print("→ 阈值越高，模型越保守，只有更高概率的样本才会被判成 1")


# ============================================================
# 4. 交叉熵：逻辑回归最常见的损失函数
# ============================================================
# 二分类交叉熵（Binary Cross Entropy, BCE）公式：
#
#   loss = -[y * log(p) + (1 - y) * log(1 - p)]
#
# 直觉：
#   - 如果真实标签 y = 1，而你给了很大的 p，loss 很小
#   - 如果真实标签 y = 1，而你给了很小的 p，loss 很大
#   - 如果真实标签 y = 0，而你给了很小的 p，loss 很小
#   - 如果真实标签 y = 0，而你给了很大的 p，loss 很大
#
# 所以交叉熵会非常惩罚“自信但错误”的预测。

print("\n" + "=" * 60)
print("4. 二分类交叉熵")
print("=" * 60)


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """计算平均二分类交叉熵。"""
    y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)
    losses = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(np.mean(losses))


positive_probability = 0.90
wrong_positive_probability = 0.10
positive_loss_good = binary_cross_entropy(np.array([1.0]), np.array([positive_probability]))
positive_loss_bad = binary_cross_entropy(np.array([1.0]), np.array([wrong_positive_probability]))

print(f"真实标签 y = 1，模型给 p = 0.90 时，loss = {positive_loss_good:.4f}")
print(f"真实标签 y = 1，模型给 p = 0.10 时，loss = {positive_loss_bad:.4f}")
print("→ 同样是真实正样本，0.10 这种自信地判错会被罚得更重")

guess_loss = binary_cross_entropy(y, guess_probabilities)
print(f"\n用猜测参数在整份训练集上的平均交叉熵 = {guess_loss:.4f}")


# ============================================================
# 5. 从零训练逻辑回归：梯度下降
# ============================================================
# 逻辑回归训练的流程和线性回归很像：
#
#   1. 先算 logit = X @ w + b
#   2. 再算 probability = sigmoid(logit)
#   3. 用交叉熵衡量当前预测有多差
#   4. 根据梯度更新参数 w, b
#
# 对逻辑回归来说，交叉熵对参数的梯度刚好比较整齐：
#
#   grad_w = X^T @ (p - y) / n
#   grad_b = mean(p - y)
#
# 这里 p 是所有样本的预测概率向量。

print("\n" + "=" * 60)
print("5. 梯度下降训练逻辑回归")
print("=" * 60)


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    n_epochs: int,
) -> tuple[np.ndarray, float, list[tuple[int, float, float]]]:
    """用梯度下降训练一个最基础的逻辑回归。"""
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0
    history: list[tuple[int, float, float]] = []

    for epoch in range(n_epochs):
        logits = X @ w + b
        probabilities = sigmoid(logits)

        grad_w = X.T @ (probabilities - y) / n_samples
        grad_b = float(np.mean(probabilities - y))

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if epoch in {0, 9, 49, 199, 999, n_epochs - 1}:
            loss = binary_cross_entropy(y, probabilities)
            preds = apply_threshold(probabilities, threshold=0.5)
            accuracy = float(np.mean(preds == y))
            history.append((epoch + 1, loss, accuracy))

    return w, b, history


trained_w, trained_b, train_history = fit_logistic_regression(
    X,
    y,
    learning_rate=0.8,
    n_epochs=2000,
)

print("训练过程快照：")
for epoch, loss, accuracy in train_history:
    print(f"  第{epoch:4d}轮: loss = {loss:.4f}, accuracy = {accuracy:.4f}")

trained_logits = X @ trained_w + trained_b
trained_probabilities = sigmoid(trained_logits)
trained_predictions = apply_threshold(trained_probabilities, threshold=0.5)

print(f"\n训练完成后的参数: w = {np.round(trained_w, 4)}, b = {trained_b:.4f}")
print("逐条样本看最终预测：")
for i, sample in enumerate(samples, start=1):
    print(
        f"  样本{i:02d}: logit = {trained_logits[i - 1]:>7.3f}, "
        f"p = {trained_probabilities[i - 1]:.4f}, "
        f"pred = {trained_predictions[i - 1]}, "
        f"true = {sample['passed']}"
    )

final_loss = binary_cross_entropy(y, trained_probabilities)
final_accuracy = np.mean(trained_predictions == y)
print(f"\n最终平均交叉熵 = {final_loss:.4f}")
print(f"最终训练集准确率 = {final_accuracy:.4f}")


# ============================================================
# 6. 训练后再看 threshold 的影响
# ============================================================
# 训练完后，概率更接近真实标签，此时再调 threshold 会更有意义。

print("\n" + "=" * 60)
print("6. 训练后再看 threshold")
print("=" * 60)

for threshold in [0.3, 0.5, 0.7]:
    preds = apply_threshold(trained_probabilities, threshold=threshold)
    accuracy = np.mean(preds == y)
    print(f"threshold = {threshold:.1f}: preds = {preds}, accuracy = {accuracy:.4f}")

print("\nlogit 的符号也可以直接帮助理解：")
print("  logit > 0  -> p > 0.5")
print("  logit = 0  -> p = 0.5")
print("  logit < 0  -> p < 0.5")


# ============================================================
# 7. 动手练习
# ============================================================

print("\n" + "=" * 60)
print("7. 动手练习")
print("=" * 60)
print("TODO(human): 把 threshold 分别改成 0.6 和 0.8，再观察哪些样本会从 1 变回 0。")
print("TODO(human): 把 mock_score / 100 这一步去掉，重新调整 learning_rate，体会特征尺度对优化速度的影响。")
