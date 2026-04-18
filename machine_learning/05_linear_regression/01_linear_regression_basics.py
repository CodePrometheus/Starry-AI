"""
机器学习基础 (5) - 线性回归
====================================
这一章解决的问题是：

1. 什么叫回归问题？
2. 线性回归里的 y = wx + b 到底是什么意思？
3. 模型如何根据数据学出合适的 w 和 b？

本脚本重点演示：
- 回归任务的定义
- 一维线性回归：用房屋面积预测房价
- 预测函数 y = wx + b
- 误差与均方误差 MSE
- 解析解（闭式解）
- 梯度下降如何逼近解析解
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 1. 准备一个最简单的回归数据集
# ============================================================
# 我们先只保留 1 个特征：房屋面积 area
# 目标值是房价 price
#
# 这就是最经典的一维回归问题：
#   输入 x：面积
#   输出 y：房价（连续值）
#
# 和分类不同，这里不是预测 "是/否"、"猫/狗"，
# 而是预测一个连续数值，所以它叫回归问题。

print("=" * 60)
print("1. 回归数据")
print("=" * 60)

areas = np.array([52, 60, 68, 74, 81, 88, 95, 104], dtype=float)
prices = np.array([320, 380, 430, 470, 520, 590, 640, 710], dtype=float)

for area, price in zip(areas, prices):
    print(f"面积 = {area:>5.1f} 平方米 -> 房价 = {price:>6.1f} 万")


# ============================================================
# 2. 线性回归模型：y = wx + b
# ============================================================
# 这是最基础的模型形式。
#
#   y_hat = wx + b
#
# 其中：
#   - x: 输入特征（这里就是面积）
#   - y_hat: 模型预测值
#   - w: 权重，控制 x 的影响有多大
#   - b: 偏置，控制整条直线整体往上还是往下平移
#
# 拿真实数据举例：
# 如果我们先随便猜：
#   w = 6
#   b = 20
#
# 那么对第 1 条样本 area = 52 来说：
#   y_hat = 6 * 52 + 20 = 332
#
# 真实房价是 320，说明这次预测高了 12。

print("\n" + "=" * 60)
print("2. 模型形式 y = wx + b")
print("=" * 60)


def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """线性回归预测函数：y_hat = wx + b"""
    return w * x + b


guess_w = 6.0
guess_b = 20.0
first_sample_pred = predict(np.array([areas[0]]), guess_w, guess_b)[0]
print(f"假设 w = {guess_w}, b = {guess_b}")
print(f"训练集第1条样本: area = {areas[0]}")
print(f"预测房价 y_hat = {guess_w} * {areas[0]} + {guess_b} = {first_sample_pred:.1f}")
print(f"真实房价 y      = {prices[0]:.1f}")
print(f"误差 y_hat - y   = {first_sample_pred - prices[0]:.1f}")


# ============================================================
# 3. 误差和损失函数：MSE
# ============================================================
# 单条样本的误差可以写成：
#   error = y_hat - y
#
# 但如果直接把所有误差加起来，正负会互相抵消，
# 所以常见做法是先平方，再求平均。
# 这就是均方误差 MSE（Mean Squared Error）。
#
# 公式：
#   MSE = (1/n) * Σ(y_hat_i - y_i)^2
#
# 为什么线性回归常用 MSE？
#   - 平方后不会正负抵消
#   - 大误差会被更重地惩罚
#   - 公式光滑，方便求导和优化

print("\n" + "=" * 60)
print("3. 均方误差 MSE")
print("=" * 60)


def compute_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """计算均方误差 MSE。"""
    return np.mean((y_pred - y_true) ** 2)


guess_preds = predict(areas, guess_w, guess_b)
guess_mse = compute_mse(guess_preds, prices)

print("用猜测参数 w = 6, b = 20 对前3条样本做预测：")
for i in range(3):
    error = guess_preds[i] - prices[i]
    sq_error = error ** 2
    print(
        f"  样本{i+1}: y_hat = {guess_preds[i]:>6.1f}, "
        f"y = {prices[i]:>6.1f}, error = {error:>6.1f}, error^2 = {sq_error:>7.1f}"
    )
print(f"\n全训练集的 MSE = {guess_mse:.3f}")


# ============================================================
# 3.5 练习题：手写 MSE
# ============================================================
# 这一题不依赖后面的章节，适合你现在立刻亲手写。
#
# 要求：
#   不要直接调用 np.mean((y_pred - y_true) ** 2)
#   而是自己拆成三步：
#   1. 先算误差 errors = y_pred - y_true
#   2. 再算平方误差 squared_errors = errors ** 2
#   3. 最后对平方误差求平均
#
# 拿当前真实数据举例：
#   如果只看前 3 条样本：
#   y_pred = [332, 380, 428]
#   y_true = [320, 380, 430]
#   errors = [12, 0, -2]
#   squared_errors = [144, 0, 4]
#   MSE = (144 + 0 + 4) / 3 = 49.333...

print("\n" + "=" * 60)
print("3.5 练习题：手写 MSE")
print("=" * 60)


def compute_mse_exercise(y_pred: np.ndarray, y_true: np.ndarray) -> float | None:
    """
    参数：
        y_pred: 模型预测值
        y_true: 真实值

    返回：
        mse: 均方误差

    提示：
        第1步：errors = y_pred - y_true
        第2步：squared_errors = errors ** 2
        第3步：mse = 平方误差的平均值

    不要直接调用上面已经写好的 compute_mse。
    """
    errors = y_pred - y_true
    squared_errors = errors ** 2
    mse = np.mean(squared_errors)
    return mse
        

exercise_preds = guess_preds[:3]
exercise_true = prices[:3]
exercise_mse = compute_mse_exercise(exercise_preds, exercise_true)
if exercise_mse is not None:
    print(f"你手写的前3条样本 MSE = {exercise_mse:.3f}")
    print(f"参考答案（函数 compute_mse） = {compute_mse(exercise_preds, exercise_true):.3f}")
else:
    print("TODO(human): 请你手写 compute_mse_exercise，完成后这里会自动显示结果。")


# ============================================================
# 4. 解析解（闭式解）
# ============================================================
# 对一维线性回归，可以直接算出最优解。
#
# 公式：
#   w = Σ[(xi - x_mean)(yi - y_mean)] / Σ[(xi - x_mean)^2]
#   b = y_mean - w * x_mean
#
# 这两个公式的直觉是：
#   - w 负责描述 x 和 y 的线性关系强度
#   - b 负责把整条线平移到合适位置
#
# 这里直接拿当前真实数据计算。

print("\n" + "=" * 60)
print("4. 解析解")
print("=" * 60)


def fit_linear_regression_closed_form(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    用一维线性回归的闭式解直接求 w 和 b。

    这两个公式是：
      w = Σ[(xi - x_mean)(yi - y_mean)] / Σ[(xi - x_mean)^2]
      b = y_mean - w * x_mean

    你可以把它拆成 4 步：
    1. 先算 x 的均值 x_mean
    2. 再算 y 的均值 y_mean
    3. 算分子 numerator = Σ[(xi - x_mean)(yi - y_mean)]
    4. 算分母 denominator = Σ[(xi - x_mean)^2]

    最后：
      w = numerator / denominator
      b = y_mean - w * x_mean
    """
    # x_mean 是所有面积的平均值。
    # 用当前真实数据：
    # x = [52, 60, 68, 74, 81, 88, 95, 104]
    # 所以：
    # x_mean = (52 + 60 + 68 + 74 + 81 + 88 + 95 + 104) / 8 = 77.75
    x_mean = np.mean(x)

    # y_mean 是所有房价的平均值。
    # 用当前真实数据：
    # y = [320, 380, 430, 470, 520, 590, 640, 710]
    # 所以：
    # y_mean = (320 + 380 + 430 + 470 + 520 + 590 + 640 + 710) / 8 = 507.5
    y_mean = np.mean(y)

    # numerator 是“x 偏离均值”和“y 偏离均值”一起变化的总量。
    # 如果 x 比平均值大时，y 也通常比平均值大，
    # 那么这个值通常会比较大。
    numerator = np.sum((x - x_mean) * (y - y_mean))

    # denominator 是 x 自己的总波动大小。
    # 它相当于问：x 围绕均值一共波动了多少。
    denominator = np.sum((x - x_mean) ** 2)

    # w 表示斜率，也就是“面积每增加 1，房价大约增加多少”。
    w = numerator / denominator

    # b 表示截距，也就是“把这条直线整体平移到合适位置”。
    b = y_mean - w * x_mean
    return w, b


closed_form_w, closed_form_b = fit_linear_regression_closed_form(areas, prices)
closed_form_preds = predict(areas, closed_form_w, closed_form_b)
closed_form_mse = compute_mse(closed_form_preds, prices)

areas_mean = np.mean(areas)
prices_mean = np.mean(prices)
closed_form_numerator = np.sum((areas - areas_mean) * (prices - prices_mean))
closed_form_denominator = np.sum((areas - areas_mean) ** 2)

print("先看闭式解里的关键中间量：")
print(f"  x_mean（面积均值） = {areas_mean:.4f}")
print(f"  y_mean（房价均值） = {prices_mean:.4f}")
print(f"  numerator（分子）  = {closed_form_numerator:.4f}")
print(f"  denominator（分母） = {closed_form_denominator:.4f}")
print(f"  w = numerator / denominator = {closed_form_numerator:.4f} / {closed_form_denominator:.4f} = {closed_form_w:.4f}")
print(f"  b = y_mean - w * x_mean = {prices_mean:.4f} - {closed_form_w:.4f} * {areas_mean:.4f} = {closed_form_b:.4f}")

print(f"解析解得到: w = {closed_form_w:.4f}, b = {closed_form_b:.4f}")
print(f"解析解对应的训练集 MSE = {closed_form_mse:.4f}")

print("\n直接拿训练集第1条样本 area = 52 做真实代入：")
print(f"  y_hat = {closed_form_w:.4f} * 52 + ({closed_form_b:.4f}) = {predict(np.array([52.0]), closed_form_w, closed_form_b)[0]:.4f}")
print(f"  真实房价 y = {prices[0]:.4f}")

print("\n解析解下前3条样本的预测：")
for i in range(3):
    print(
        f"  area = {areas[i]:>5.1f}, "
        f"y_hat = {closed_form_preds[i]:>6.2f}, "
        f"y = {prices[i]:>6.2f}"
    )


# ============================================================
# 5. 梯度下降求解线性回归
# ============================================================
# 前面第 02 章我们已经学过：
#   参数 = 参数 - 学习率 * 梯度
#
# 现在把它真正用在线性回归里。
#
# MSE 对参数的梯度：
#   dL/dw = (2/n) * Σ[(y_hat_i - y_i) * x_i]
#   dL/db = (2/n) * Σ[(y_hat_i - y_i)]
#
# 这两个公式不是凭空来的，可以先从 1 条样本推。
#
# 只看单条样本时：
#   L = (y_hat - y)^2
#   y_hat = wx + b
# 所以：
#   L = (wx + b - y)^2
#
# 对 w 求导：
#   dL/dw = 2(wx + b - y) * x
# 这里会多出一个 x，
# 因为 w 是乘在 x 前面的，所以对 w 求导时 x 会保留下来。
#
# 对 b 求导：
#   dL/db = 2(wx + b - y)
# 这里不会多出 x，
# 因为 b 是直接加上去的，对 b 求导只会留下 1。
#
# 再把 1 条样本推广到 n 条样本并求平均，
# 就得到整批数据的梯度：
#   dL/dw = (2/n) * Σ[(y_hat_i - y_i) * x_i]
#   dL/db = (2/n) * Σ[(y_hat_i - y_i)]
#
# 直接拿当前真实数据举例：
#   训练集第1条样本：
#   x = 52, y = 320
#
#   如果当前参数先假设：
#   w = 6, b = 20
#
#   那么：
#   y_hat = 6 * 52 + 20 = 332
#   y_hat - y = 332 - 320 = 12
#
#   这一条样本对 w 的梯度贡献：
#   2 * (y_hat - y) * x = 2 * 12 * 52 = 1248
#
#   这一条样本对 b 的梯度贡献：
#   2 * (y_hat - y) = 2 * 12 = 24
#
# 所以你可以直观看到：
#   dL/dw 比 dL/db 多一个 x，
#   因为 w 和输入 x 是绑在一起的。
#
# 这两个公式的总体含义：
#   - 如果预测普遍偏大，梯度会推动参数往减小 loss 的方向更新
#   - 如果预测普遍偏小，梯度会推动参数往增大 loss 的方向更新

print("\n" + "=" * 60)
print("5. 梯度下降")
print("=" * 60)


def fit_linear_regression_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    n_steps: int,
) -> tuple[float, float, list[tuple[int, float, float, float]]]:
    """
    用梯度下降拟合一维线性回归。

    这个函数的工作流程可以直接理解成 4 步循环：
    1. 用当前的 w 和 b 去预测 y_hat
    2. 把预测值和真实值比较，得到误差 errors
    3. 根据误差计算 w 和 b 的梯度
    4. 按“参数 = 参数 - 学习率 * 梯度”去更新参数

    为什么不断重复这 4 步，最后会得到比较合理的 w 和 b？
    因为梯度在告诉我们：如果当前 loss 想继续下降，参数应该朝哪个方向改。
    只要学习率合适，每一步都会朝“让 MSE 变小”的方向走一点。

    直接拿当前真实数据看第 1 步的直觉：
    - 初始时 w = 0, b = 0
    - 所有样本都会被预测成 0
    - 真实房价却是 [320, 380, 430, ...]
    - 这说明模型严重低估
    - 所以梯度会推动参数往“提高预测值”的方向改

    也就是说：
    一开始模型很差，但它不是瞎改参数，
    而是每一步都根据“当前错了多少”来系统地修正自己。
    """
    w = 0.0
    b = 0.0
    n = len(x)
    history = []

    for step in range(n_steps):
        # 第1步：用当前参数做预测。
        # 这里 y_pred 和 x 形状相同，表示每条样本当前的预测房价。
        y_pred = predict(x, w, b)

        # 第2步：计算误差。
        # errors[i] = 第 i 条样本的预测值 - 真实值
        # 如果 error > 0，说明预测偏大
        # 如果 error < 0，说明预测偏小
        errors = y_pred - y

        # 第3步：计算梯度。
        #
        # grad_w 关心的是：
        #   “如果我改动 w，loss 会怎么变？”
        # 因为 w 是乘在 x 前面的，所以 grad_w 会乘上 x。
        #
        # grad_b 关心的是：
        #   “如果我改动 b，loss 会怎么变？”
        # 因为 b 是直接加上的，所以 grad_b 不会乘 x。
        grad_w = (2 / n) * np.sum(errors * x)
        grad_b = (2 / n) * np.sum(errors)

        # 第4步：更新参数。
        # 沿着负梯度方向走一小步，让 loss 下降。
        # 学习率 learning_rate 控制每步走多远。
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        # 这里只挑若干关键步记录下来，避免打印 5000 行。
        if step in [0, 1, 2, 9, 99, 999, n_steps - 1]:
            mse = compute_mse(predict(x, w, b), y)
            history.append((step + 1, w, b, mse))

    return w, b, history


gd_w, gd_b, gd_history = fit_linear_regression_gradient_descent(
    areas,
    prices,
    learning_rate=1e-4,
    n_steps=5000,
)
gd_preds = predict(areas, gd_w, gd_b)
gd_mse = compute_mse(gd_preds, prices)

print("梯度下降关键步骤：")
for step, w, b, mse in gd_history:
    print(f"  第{step:4d}步: w = {w:8.4f}, b = {b:8.4f}, MSE = {mse:10.4f}")

print(f"\n梯度下降最终得到: w = {gd_w:.4f}, b = {gd_b:.4f}")
print(f"梯度下降对应的训练集 MSE = {gd_mse:.4f}")


# ============================================================
# 6. 解析解 vs 梯度下降
# ============================================================
# 对线性回归来说，解析解和梯度下降都能求参数。
# 区别在于：
#   - 解析解：一步算出答案（适合简单情况）
#   - 梯度下降：迭代逼近答案（更通用，后面深度学习会大量用它）
#
# 这里我们把两种结果放一起看。

print("\n" + "=" * 60)
print("6. 两种求解方式对比")
print("=" * 60)

print(f"解析解:   w = {closed_form_w:.4f}, b = {closed_form_b:.4f}, MSE = {closed_form_mse:.4f}")
print(f"梯度下降: w = {gd_w:.4f}, b = {gd_b:.4f}, MSE = {gd_mse:.4f}")

comparison_area = 90.0
closed_form_prediction = predict(np.array([comparison_area]), closed_form_w, closed_form_b)[0]
gd_prediction = predict(np.array([comparison_area]), gd_w, gd_b)[0]

print(f"\n拿真实面积 area = {comparison_area:.1f} 举例：")
print(f"  解析解预测房价 = {closed_form_prediction:.2f}")
print(f"  梯度下降预测房价 = {gd_prediction:.2f}")


# ============================================================
# 7. 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
这一章的核心链路：

1. 线性回归要解决的是“预测连续值”问题
2. 模型形式是 y_hat = wx + b
3. w 控制斜率，b 控制整体平移
4. 常见损失函数是 MSE
5. 线性回归既可以用解析解求，也可以用梯度下降求

你现在应该把线性回归理解成：
“最基础的监督学习回归模型，也是后面更复杂模型的起点。”
""")
