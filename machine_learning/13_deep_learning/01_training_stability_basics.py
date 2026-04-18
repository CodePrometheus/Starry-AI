"""
深度学习基础 (13) - 训练稳定性
================================
这一章解决的问题是：

1. 初始化为什么会影响训练能不能顺利开始？
2. 学习率太大、太小分别会发生什么？
3. batch size 为什么会影响梯度噪声和训练手感？
4. 过拟合出现时，正则化为什么能帮忙？
"""

from __future__ import annotations

import numpy as np


np.random.seed(42)


def tanh(x: np.ndarray) -> np.ndarray:
    """双曲正切激活函数。"""
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活。"""
    return np.maximum(0.0, x)


print("=" * 60)
print("1. 初始化：太小、太大、合适有什么区别？")
print("=" * 60)

# 用一个小型用户行为特征矩阵观察初始化后的激活分布。
# 每一行是一位用户：
#   - session_minutes: 一次会话时长（分钟）
#   - clicked_items: 点击商品数
#   - collected_items: 收藏商品数
#   - history_orders: 历史订单数
X_init_demo = np.array(
    [
        [12, 3, 1, 0],
        [25, 5, 2, 1],
        [38, 8, 5, 3],
        [52, 10, 6, 4],
        [16, 2, 0, 0],
        [44, 9, 4, 2],
    ],
    dtype=float,
)

# 训练前一般会先标准化，避免不同量纲差太大。
X_init_demo = (X_init_demo - X_init_demo.mean(axis=0)) / X_init_demo.std(axis=0)


def analyze_initialization(name: str, weight_scale: float) -> None:
    """对比不同初始化尺度下，隐藏层激活是否容易饱和。"""
    W1 = np.random.randn(X_init_demo.shape[1], 6) * weight_scale
    b1 = np.zeros(6, dtype=float)
    Z1 = X_init_demo @ W1 + b1
    A1 = tanh(Z1)

    activation_std = A1.std()
    saturation_ratio = np.mean(np.abs(A1) > 0.95)

    print(f"\n{name}")
    print(f"  权重尺度 = {weight_scale}")
    print(f"  Z1 的均值/标准差 = {Z1.mean():.4f} / {Z1.std():.4f}")
    print(f"  A1 的均值/标准差 = {A1.mean():.4f} / {activation_std:.4f}")
    print(f"  激活饱和比例(|tanh(z)| > 0.95) = {saturation_ratio:.2%}")
    print(f"  第一条样本隐藏层输出 = {np.round(A1[0], 4)}")


analyze_initialization("初始化过小", weight_scale=0.01)
analyze_initialization("初始化过大", weight_scale=3.0)
analyze_initialization("较合理的 Xavier 量级", weight_scale=np.sqrt(1 / X_init_demo.shape[1]))

print("\n结论：")
print("  初始化过小 -> 各层信号接近 0，不容易拉开样本差异。")
print("  初始化过大 -> tanh 很容易饱和到 -1 或 1，梯度会变差。")
print("  合适初始化 -> 激活分布更平衡，训练更容易开始。")


print("\n" + "=" * 60)
print("2. 学习率：步子太小、合适、太大")
print("=" * 60)

# 用标准化后的一维回归演示学习率差异。
# 任务：用门店营业面积预测月营业额（单位：万元）
shop_area = np.array([38, 45, 52, 61, 73, 81, 90, 104, 118, 126], dtype=float)
shop_revenue = np.array([21, 24, 29, 34, 41, 47, 52, 61, 68, 73], dtype=float)

x_lr = (shop_area - shop_area.mean()) / shop_area.std()
y_lr = (shop_revenue - shop_revenue.mean()) / shop_revenue.std()


def train_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    steps: int,
) -> tuple[float, float, list[float]]:
    """用全量梯度下降训练 y = wx + b。"""
    w = 0.0
    b = 0.0
    loss_history: list[float] = []

    n = len(x)
    for _ in range(steps):
        y_pred = w * x + b
        error = y_pred - y
        loss = np.mean(error ** 2)
        loss_history.append(loss)

        grad_w = (2.0 / n) * np.sum(error * x)
        grad_b = (2.0 / n) * np.sum(error)

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return w, b, loss_history


lr_settings = [0.01, 0.2, 1.2]
for lr in lr_settings:
    w_lr, b_lr, loss_history_lr = train_linear_regression(x_lr, y_lr, learning_rate=lr, steps=20)
    print(f"\nlr = {lr}")
    print(f"  第 1 步 loss = {loss_history_lr[0]:.6f}")
    print(f"  第 5 步 loss = {loss_history_lr[4]:.6f}")
    print(f"  第10步 loss = {loss_history_lr[9]:.6f}")
    print(f"  第20步 loss = {loss_history_lr[-1]:.6f}")
    print(f"  最终参数 w={w_lr:.4f}, b={b_lr:.4f}")

print("\n一般现象是：")
print("  lr 太小 -> loss 会下降，但很慢。")
print("  lr 合适 -> loss 稳定下降。")
print("  lr 太大 -> loss 容易震荡，甚至直接发散。")


print("\n" + "=" * 60)
print("3. Batch Size：为什么小批量梯度更“吵”？")
print("=" * 60)

# 这里不重新训练模型，只固定在某一个参数点，
# 观察不同 batch size 算出来的梯度抖动有多大。
current_w = 0.45
current_b = -0.10


def batch_gradient(
    x: np.ndarray,
    y: np.ndarray,
    w: float,
    b: float,
    indices: np.ndarray,
) -> tuple[float, float]:
    """计算一个 batch 上的梯度。"""
    x_batch = x[indices]
    y_batch = y[indices]
    error = (w * x_batch + b) - y_batch
    grad_w = 2.0 * np.mean(error * x_batch)
    grad_b = 2.0 * np.mean(error)
    return grad_w, grad_b


def inspect_batch_noise(batch_size: int, repeats: int = 8) -> None:
    """重复采样多个 batch，比较梯度估计的波动。"""
    grads = []
    for _ in range(repeats):
        indices = np.random.choice(len(x_lr), size=batch_size, replace=False)
        grads.append(batch_gradient(x_lr, y_lr, current_w, current_b, indices))

    grads_array = np.array(grads)
    print(f"\nbatch_size = {batch_size}")
    print(f"  前 3 次采样得到的梯度 = {np.round(grads_array[:3], 4)}")
    print(
        f"  grad_w 的均值/标准差 = "
        f"{grads_array[:, 0].mean():.4f} / {grads_array[:, 0].std():.4f}"
    )
    print(
        f"  grad_b 的均值/标准差 = "
        f"{grads_array[:, 1].mean():.4f} / {grads_array[:, 1].std():.4f}"
    )


inspect_batch_noise(batch_size=1)
inspect_batch_noise(batch_size=4)
inspect_batch_noise(batch_size=len(x_lr))

print("\n直觉上：")
print("  小 batch 便宜、更新快，但梯度方差更大。")
print("  大 batch 更稳，但一次更新更贵，也更像“平均意见”。")


print("\n" + "=" * 60)
print("4. 过拟合与正则化：高阶多项式为什么会记住训练集？")
print("=" * 60)

# 用一条带噪声的真实曲线做例子：
#   x = 广告投放强度（归一化）
#   y = 转化率
#
# 我们故意只给很少的训练样本，这样高阶模型很容易把噪声也记进去。
x_train_curve = np.array([-1.2, -0.9, -0.5, -0.1, 0.2, 0.45, 0.7, 1.0], dtype=float)
y_train_curve = np.array([-0.62, -0.71, -0.40, -0.06, 0.22, 0.58, 0.71, 0.88], dtype=float)

x_valid_curve = np.array([-1.1, -0.7, -0.3, 0.0, 0.3, 0.6, 0.9, 1.1], dtype=float)
y_valid_curve = np.array([-0.70, -0.55, -0.21, 0.02, 0.28, 0.56, 0.79, 0.93], dtype=float)


def make_polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """把一维输入扩展成多项式特征 [1, x, x^2, ...]。"""
    return np.column_stack([x ** power for power in range(degree + 1)])


def fit_ridge_regression(X: np.ndarray, y: np.ndarray, l2_lambda: float) -> np.ndarray:
    """闭式解求带 L2 正则的线性回归参数。"""
    identity = np.eye(X.shape[1], dtype=float)
    identity[0, 0] = 0.0  # 不对偏置项做正则
    return np.linalg.solve(X.T @ X + l2_lambda * identity, X.T @ y)


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """均方误差。"""
    return np.mean((y_pred - y_true) ** 2)


def evaluate_polynomial_model(degree: int, l2_lambda: float) -> None:
    """训练并打印训练集/验证集误差。"""
    X_train_poly = make_polynomial_features(x_train_curve, degree=degree)
    X_valid_poly = make_polynomial_features(x_valid_curve, degree=degree)

    coef = fit_ridge_regression(X_train_poly, y_train_curve, l2_lambda=l2_lambda)
    train_pred = X_train_poly @ coef
    valid_pred = X_valid_poly @ coef

    train_mse = mse(train_pred, y_train_curve)
    valid_mse = mse(valid_pred, y_valid_curve)
    coef_norm = np.linalg.norm(coef[1:])

    print(f"\n多项式次数 = {degree}, L2 = {l2_lambda}")
    print(f"  训练集 MSE = {train_mse:.6f}")
    print(f"  验证集 MSE = {valid_mse:.6f}")
    print(f"  参数范数(不含偏置) = {coef_norm:.4f}")
    print(f"  前 5 个系数 = {np.round(coef[:5], 4)}")


evaluate_polynomial_model(degree=2, l2_lambda=0.0)
evaluate_polynomial_model(degree=8, l2_lambda=0.0)
evaluate_polynomial_model(degree=8, l2_lambda=1.0)

print("\n观察重点：")
print("  高阶模型 + 无正则 时，训练误差可能很低，但验证误差变差。")
print("  加上 L2 正则后，参数被压小，验证误差往往更稳。")


print("\n" + "=" * 60)
print("5. 练习题")
print("=" * 60)
print("TODO(human): 把初始化实验再补一个 He 初始化（适合 ReLU），")
print("并把第 4 部分的 L2 强度从 1.0 改成 0.1、10.0 各跑一次，比较验证误差。")
