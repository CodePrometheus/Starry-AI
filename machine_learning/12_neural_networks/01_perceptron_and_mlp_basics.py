"""
机器学习基础 (12) - 感知机与 MLP 入门
======================================
这一章解决的问题是：

1. 单个神经元到底在算什么？
2. 感知机为什么本质上是“线性分类器”？
3. 多层感知机（MLP）为什么能表达更复杂的模式？
4. 前向传播里的矩阵乘法到底对应了什么计算？
"""

from __future__ import annotations

import numpy as np


np.random.seed(42)


def step(x: np.ndarray) -> np.ndarray:
    """阶跃激活：大于等于 0 输出 1，否则输出 0。"""
    return (x >= 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 激活，常用于二分类输出层。"""
    return 1.0 / (1.0 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活：负数截断为 0，正数保持不变。"""
    return np.maximum(0.0, x)


print("=" * 60)
print("1. 单个神经元：加权求和 + 激活函数")
print("=" * 60)

# 这里用一个非常具体的业务例子：
# 判断一个实习生候选人是否值得进入终面。
# 我们只看两个已经量化过的特征：
#   - coding_test: 编程笔试分（满分 10）
#   - project_demo: 项目讲解表现（满分 10）
#
# 标签：
#   1 = 建议进入终面
#   0 = 暂时不进入终面
candidate_samples = [
    {"name": "小李", "coding_test": 9, "project_demo": 8, "label": 1},
    {"name": "小王", "coding_test": 8, "project_demo": 7, "label": 1},
    {"name": "小陈", "coding_test": 7, "project_demo": 4, "label": 0},
    {"name": "小赵", "coding_test": 4, "project_demo": 8, "label": 0},
    {"name": "小周", "coding_test": 3, "project_demo": 3, "label": 0},
    {"name": "小孙", "coding_test": 8, "project_demo": 9, "label": 1},
]

X_candidates = np.array(
    [[row["coding_test"], row["project_demo"]] for row in candidate_samples],
    dtype=float,
)
y_candidates = np.array([row["label"] for row in candidate_samples], dtype=float)

# 感知机的核心公式：
#   z = w1*x1 + w2*x2 + b
#   y_hat = step(z)
#
# 直觉上：
#   - 权重 w 控制每个特征的重要性
#   - 偏置 b 控制决策边界整体平移
#   - step(z) 把连续分数变成 0/1 决策
w_perceptron = np.array([0.8, 0.6], dtype=float)
b_perceptron = -10.0

z_candidates = X_candidates @ w_perceptron + b_perceptron
y_pred_candidates = step(z_candidates)

print(f"感知机权重 w = {w_perceptron}")
print(f"感知机偏置 b = {b_perceptron}")
print("公式: z = 0.8*coding_test + 0.6*project_demo - 10")
print()

for row, z_value, pred in zip(candidate_samples, z_candidates, y_pred_candidates):
    print(
        f"{row['name']}: coding={row['coding_test']}, demo={row['project_demo']}, "
        f"z={z_value:6.2f}, 预测={int(pred)}, 真实={row['label']}"
    )

candidate_accuracy = np.mean(y_pred_candidates == y_candidates)
print(f"\n训练样本上的分类准确率 = {candidate_accuracy:.2%}")
print("→ 单个感知机做的事，本质上就是一条直线把平面切成两边。")

first_x = X_candidates[0]
first_z = z_candidates[0]
print("\n拿第一条样本“小李”手算一次：")
print(f"  输入 x = {first_x}")
print(f"  z = 0.8*{first_x[0]} + 0.6*{first_x[1]} - 10 = {first_z:.2f}")
print(f"  step(z) = {int(y_pred_candidates[0])}")


print("\n" + "=" * 60)
print("2. 感知机的局限：为什么 XOR 搞不定？")
print("=" * 60)

# XOR 是理解 MLP 的经典例子。
# 这里给它换成更贴近日常的解释：
#   x1 = 今天是不是周末
#   x2 = 用户手里有没有优惠券
# 目标：
#   只有“恰好满足一个条件”时，用户才会冲动下单
#   两个都没有 → 不下单
#   两个都有   → 反而会犹豫（这里刻意构造 XOR 模式）
xor_feature_names = ["is_weekend", "has_coupon"]
xor_samples = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=float,
)
xor_labels = np.array([0, 1, 1, 0], dtype=float)

# 我们先随便给一个单层感知机参数，看看它最多只能画一条线。
w_xor = np.array([1.0, 1.0], dtype=float)
b_xor = -0.5
z_xor = xor_samples @ w_xor + b_xor
y_xor_pred = step(z_xor)

print(f"单层感知机参数: w = {w_xor}, b = {b_xor}")
for sample, z_value, pred, label in zip(xor_samples, z_xor, y_xor_pred, xor_labels):
    print(
        f"x={sample.astype(int)} -> z={z_value:5.1f}, "
        f"预测={int(pred)}, 真实={int(label)}"
    )

print("\n观察四个点的位置：")
print("  [0,1] 和 [1,0] 应该分到正类")
print("  [0,0] 和 [1,1] 应该分到负类")
print("→ 这四个点没法用一条直线正确切开，所以单个感知机不够。")


print("\n" + "=" * 60)
print("3. 两层 MLP：隐藏层先造特征，再做输出")
print("=" * 60)

# 现在我们手工构造一个 2 层网络来解决 XOR。
#
# 隐藏层神经元 1：
#   h1 = step(x1 + x2 - 0.5)
#   只要两个条件里有一个满足，它就亮起（近似 OR）
#
# 隐藏层神经元 2：
#   h2 = step(x1 + x2 - 1.5)
#   只有两个条件都满足，它才亮起（近似 AND）
#
# 输出层：
#   y = step(1*h1 - 2*h2 - 0.5)
#   逻辑相当于：OR 但不要 AND，也就是 XOR
W1_xor = np.array(
    [
        [1.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=float,
)
b1_xor = np.array([-0.5, -1.5], dtype=float)
W2_xor = np.array([[1.0], [-2.0]], dtype=float)
b2_xor = np.array([-0.5], dtype=float)

Z1_xor = xor_samples @ W1_xor + b1_xor
A1_xor = step(Z1_xor)
Z2_xor = A1_xor @ W2_xor + b2_xor
y_xor_mlp = step(Z2_xor).reshape(-1)

print("隐藏层输出 A1：")
print(A1_xor.astype(int))
print("\n输出层原始分数 Z2：")
print(Z2_xor.reshape(-1))

for sample, hidden, pred, label in zip(xor_samples, A1_xor, y_xor_mlp, xor_labels):
    print(
        f"x={sample.astype(int)} -> hidden={hidden.astype(int)} -> "
        f"预测={int(pred)}, 真实={int(label)}"
    )

print("\n拿 x=[1,0] 手算一次完整前向传播：")
manual_x = np.array([[1.0, 0.0]])
manual_z1 = manual_x @ W1_xor + b1_xor
manual_a1 = step(manual_z1)
manual_z2 = manual_a1 @ W2_xor + b2_xor
manual_y = step(manual_z2)
print(f"  输入 x = {manual_x}")
print(f"  Z1 = x @ W1 + b1 = {manual_z1}")
print(f"  A1 = step(Z1)   = {manual_a1}")
print(f"  Z2 = A1 @ W2 + b2 = {manual_z2}")
print(f"  y_hat = step(Z2)  = {manual_y.astype(int)}")
print("→ 隐藏层相当于先学出“至少满足一个条件”和“两个条件都满足”这两个中间特征。")


print("\n" + "=" * 60)
print("4. 前向传播直觉：批量样本如何做矩阵乘法")
print("=" * 60)

# 真实训练时，不会一条一条样本算，而是把一批样本拼成矩阵一起算。
# 下面用 3 个候选人样本演示：
#
#   X      形状 = (3, 2)  -> 3条样本, 每条 2个特征
#   W1     形状 = (2, 3)  -> 输入层到隐藏层
#   b1     形状 = (3,)    -> 3个隐藏神经元各一个偏置
#   W2     形状 = (3, 1)  -> 隐藏层到输出层
X_batch = X_candidates[:3]
W1_batch = np.array(
    [
        [0.9, -0.3, 0.6],
        [0.5, 0.8, -0.2],
    ],
    dtype=float,
)
b1_batch = np.array([-7.5, -2.0, -3.0], dtype=float)
W2_batch = np.array([[1.2], [-0.7], [0.9]], dtype=float)
b2_batch = np.array([-1.0], dtype=float)

Z1_batch = X_batch @ W1_batch + b1_batch
A1_batch = relu(Z1_batch)
Z2_batch = A1_batch @ W2_batch + b2_batch
A2_batch = sigmoid(Z2_batch)

print(f"输入矩阵 X_batch 形状 = {X_batch.shape}")
print(X_batch)
print(f"\n第一层权重 W1 形状 = {W1_batch.shape}")
print(W1_batch)
print(f"\n第一层线性输出 Z1 = X @ W1 + b1，形状 = {Z1_batch.shape}")
print(np.round(Z1_batch, 3))
print(f"\n隐藏层激活 A1 = ReLU(Z1)，形状 = {A1_batch.shape}")
print(np.round(A1_batch, 3))
print(f"\n输出层线性结果 Z2，形状 = {Z2_batch.shape}")
print(np.round(Z2_batch, 3))
print(f"\n最终概率 A2 = sigmoid(Z2)，形状 = {A2_batch.shape}")
print(np.round(A2_batch, 4))

print("\n看第一条样本是怎么映射到第一个隐藏神经元的：")
print(
    "  Z1[0,0] = "
    f"{X_batch[0,0]}*{W1_batch[0,0]} + {X_batch[0,1]}*{W1_batch[1,0]} + {b1_batch[0]}"
    f" = {Z1_batch[0,0]:.3f}"
)
print("→ 这就是“当前样本的所有输入特征”和“这个神经元自己的权重列”做一次点积。")


print("\n" + "=" * 60)
print("5. 练习题")
print("=" * 60)
print("TODO(human): 把 XOR 例子里的隐藏层从 step 改成 sigmoid，")
print("然后自己手动调整 W1/W2，观察前向传播输出如何变化。")
