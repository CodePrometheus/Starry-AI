"""
机器学习基础 (11) - 模型评估基础
====================================
这一章解决的问题是：

1. train / valid / test 各自负责什么？
2. 回归任务和分类任务分别该看哪些指标？
3. confusion matrix、precision、recall、F1 应该怎么理解？
4. 什么叫数据泄漏？为什么实验记录很重要？

本脚本重点演示：
- 训练集 / 验证集 / 测试集的职责分工
- 回归指标：MAE、MSE、RMSE、R²
- 分类指标：accuracy、confusion matrix、precision、recall、F1
- 数据泄漏与实验记录的基本意识
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 1. train / valid / test：每一份数据负责什么
# ============================================================
# 我们继续用一个简单的回归场景：
#   输入 x: 每天学习时长（小时）
#   输出 y: 模拟考试成绩
#
# 为了突出评估逻辑，这里直接手动切分：
#   - train: 拿来学参数
#   - valid: 拿来选模型
#   - test : 最后一次客观评估
#
# 注意：
#   test 不应该在调模型阶段反复偷看。
#   否则 test 就不再是“真正未知的数据”了。

print("=" * 60)
print("1. train / valid / test 的角色")
print("=" * 60)

x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
y_train = np.array([52.0, 55.0, 60.0, 65.0, 68.0, 72.0], dtype=float)

x_valid = np.array([2.5, 4.5, 6.5], dtype=float)
y_valid = np.array([57.0, 66.0, 75.0], dtype=float)

x_test = np.array([3.5, 5.5, 7.0], dtype=float)
y_test = np.array([62.0, 70.0, 78.0], dtype=float)

print(f"训练集 x_train = {x_train}, y_train = {y_train}")
print(f"验证集 x_valid = {x_valid}, y_valid = {y_valid}")
print(f"测试集 x_test  = {x_test}, y_test  = {y_test}")


def fit_mean_regressor(y: np.ndarray) -> float:
    """最简单的 baseline：永远预测训练集均值。"""
    return float(y.mean())


def predict_mean_regressor(x: np.ndarray, mean_value: float) -> np.ndarray:
    """给任意输入都返回同一个均值预测。"""
    return np.full(shape=x.shape, fill_value=mean_value, dtype=float)


def fit_linear_regression_closed_form(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """一维线性回归解析解。"""
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    w = numerator / denominator
    b = y_mean - w * x_mean
    return float(w), float(b)


def predict_linear_regression(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """线性回归预测。"""
    return w * x + b


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE：平均绝对误差。"""
    return float(np.mean(np.abs(y_true - y_pred)))


mean_baseline = fit_mean_regressor(y_train)
baseline_valid_pred = predict_mean_regressor(x_valid, mean_baseline)

linear_w, linear_b = fit_linear_regression_closed_form(x_train, y_train)
linear_valid_pred = predict_linear_regression(x_valid, linear_w, linear_b)

baseline_valid_mae = mean_absolute_error(y_valid, baseline_valid_pred)
linear_valid_mae = mean_absolute_error(y_valid, linear_valid_pred)

print(f"\n均值 baseline 在训练集学到的常数 = {mean_baseline:.3f}")
print(f"线性回归学到的参数: w = {linear_w:.3f}, b = {linear_b:.3f}")
print(f"\n验证集上 baseline 的 MAE = {baseline_valid_mae:.3f}")
print(f"验证集上线性回归的 MAE = {linear_valid_mae:.3f}")

if linear_valid_mae < baseline_valid_mae:
    selected_model_name = "linear_regression"
    selected_test_pred = predict_linear_regression(x_test, linear_w, linear_b)
else:
    selected_model_name = "mean_baseline"
    selected_test_pred = predict_mean_regressor(x_test, mean_baseline)

print(f"\n根据验证集表现，选择的模型是: {selected_model_name}")
print("到这一步之前，我们都不应该根据 test 的结果来改模型。")


# ============================================================
# 2. 回归任务常见指标
# ============================================================
# 下面只在“最终选中的模型”上看 test 指标。
# 这是一个更接近真实流程的做法：
#   先用 valid 选模型
#   再用 test 报告最终泛化效果

print("\n" + "=" * 60)
print("2. 回归指标：MAE / MSE / RMSE / R²")
print("=" * 60)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE：平均平方误差。"""
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE：MSE 开平方，量纲和原目标一致。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R²：模型解释了多少相对基线的方差。"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


test_mae = mean_absolute_error(y_test, selected_test_pred)
test_mse = mean_squared_error(y_test, selected_test_pred)
test_rmse = root_mean_squared_error(y_test, selected_test_pred)
test_r2 = r2_score(y_test, selected_test_pred)

print(f"测试集真实值 y_test      = {y_test}")
print(f"测试集预测值 y_test_pred = {np.round(selected_test_pred, 3)}")
print(f"\nMAE  = {test_mae:.4f}")
print(f"MSE  = {test_mse:.4f}")
print(f"RMSE = {test_rmse:.4f}")
print(f"R²   = {test_r2:.4f}")

print("\n怎么理解这些指标？")
print("MAE 更像“平均差了多少分”；")
print("MSE / RMSE 会更重地惩罚大误差；")
print("R² 越接近 1，说明模型越能解释成绩变化。")


# ============================================================
# 3. 分类任务：confusion matrix / precision / recall / F1
# ============================================================
# 现在切换到一个分类场景：
#   预测订单是否为欺诈订单（1 = 欺诈, 0 = 正常）
#
# 欺诈订单通常占比很低，所以不能只看 accuracy。
# 一个“永远预测正常”的模型，accuracy 也可能看起来不差。

print("\n" + "=" * 60)
print("3. 分类指标")
print("=" * 60)

y_true_cls = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=int)
y_score_cls = np.array([0.91, 0.40, 0.15, 0.05, 0.62, 0.30, 0.18, 0.55, 0.48, 0.10, 0.08, 0.20], dtype=float)

print(f"真实标签 y_true_cls   = {y_true_cls}")
print(f"模型输出概率 y_score = {y_score_cls}")


def predict_by_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """把概率转成 0/1 预测。"""
    return (scores >= threshold).astype(int)


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """返回 [[TN, FP], [FN, TP]]。"""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """分类准确率。"""
    return float(np.mean(y_true == y_pred))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision：预测成正类的样本里，真的有多少是正类。"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denominator = tp + fp
    return float(tp / denominator) if denominator != 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall：所有真正的正类里，抓到了多少。"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denominator = tp + fn
    return float(tp / denominator) if denominator != 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1：precision 和 recall 的调和平均。"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    denominator = precision + recall
    return float(2 * precision * recall / denominator) if denominator != 0 else 0.0


for threshold in [0.5, 0.4]:
    y_pred_cls = predict_by_threshold(y_score_cls, threshold)
    cm = confusion_matrix_binary(y_true_cls, y_pred_cls)
    acc = accuracy_score(y_true_cls, y_pred_cls)
    precision = precision_score(y_true_cls, y_pred_cls)
    recall = recall_score(y_true_cls, y_pred_cls)
    f1 = f1_score(y_true_cls, y_pred_cls)

    print(f"\n阈值 threshold = {threshold:.1f}")
    print(f"预测标签 y_pred = {y_pred_cls}")
    print(f"confusion matrix =\n{cm}")
    print(f"accuracy  = {acc:.4f}")
    print(f"precision = {precision:.4f}")
    print(f"recall    = {recall:.4f}")
    print(f"F1        = {f1:.4f}")

print("\n在这个例子里，把阈值从 0.5 降到 0.4：")
print("召回率 recall 上升了，因为抓到了更多欺诈订单；")
print("但 precision 会下降，因为误报也变多了。")


# ============================================================
# 4. 数据泄漏：为什么“看起来特别准”反而可疑
# ============================================================
# 数据泄漏的本质是：
# 训练或评估阶段不小心用了“未来信息”或“答案的变体”。
#
# 下面这个表格里，我们想预测订单是否欺诈。
# 但有一个字段 chargeback_days_later：
#   - 正常订单记为 0
#   - 欺诈订单在几天后被拒付，就记具体天数
#
# 这个字段只有在订单发生并且后续拒付之后才知道。
# 如果拿它做特征，模型当然会异常准确，但那不是“学会了预测”，
# 而是“偷看了未来”。

print("\n" + "=" * 60)
print("4. 数据泄漏示例")
print("=" * 60)

fraud_orders = [
    {"order_amount": 38, "night_order": 0, "chargeback_days_later": 0, "is_fraud": 0},
    {"order_amount": 420, "night_order": 1, "chargeback_days_later": 12, "is_fraud": 1},
    {"order_amount": 55, "night_order": 0, "chargeback_days_later": 0, "is_fraud": 0},
    {"order_amount": 310, "night_order": 1, "chargeback_days_later": 9, "is_fraud": 1},
    {"order_amount": 72, "night_order": 0, "chargeback_days_later": 0, "is_fraud": 0},
    {"order_amount": 260, "night_order": 1, "chargeback_days_later": 0, "is_fraud": 0},
]

for row in fraud_orders:
    print(row)

leaky_pred = np.array([1 if row["chargeback_days_later"] > 0 else 0 for row in fraud_orders], dtype=int)
safe_rule_pred = np.array(
    [1 if (row["order_amount"] > 250 and row["night_order"] == 1) else 0 for row in fraud_orders],
    dtype=int,
)
fraud_true = np.array([row["is_fraud"] for row in fraud_orders], dtype=int)

print(f"\n泄漏特征规则的预测 = {leaky_pred}")
print(f"安全规则的预测     = {safe_rule_pred}")
print(f"真实标签           = {fraud_true}")
print(f"泄漏规则 accuracy  = {accuracy_score(fraud_true, leaky_pred):.4f}")
print(f"安全规则 accuracy  = {accuracy_score(fraud_true, safe_rule_pred):.4f}")
print("\n看到一个几乎完美的结果时，不要先激动，先检查：")
print("这个特征是不是在真正预测时根本拿不到。")


# ============================================================
# 5. 实验记录：不要只记“哪个模型最好”
# ============================================================
# 一个最小可用的实验记录，至少要写下：
#   - 模型名
#   - 用了什么特征 / 数据切分
#   - 在 valid 上的结果
#   - 为什么保留或淘汰

print("\n" + "=" * 60)
print("5. 实验记录示例")
print("=" * 60)

experiment_log = [
    {
        "name": "mean_baseline",
        "split": "study_hours_v1",
        "valid_mae": round(baseline_valid_mae, 4),
        "decision": "保留做 baseline，对照用",
    },
    {
        "name": "linear_regression",
        "split": "study_hours_v1",
        "valid_mae": round(linear_valid_mae, 4),
        "decision": "验证集更好，进入最终 test",
    },
]

for experiment in experiment_log:
    print(experiment)

print("\n实验记录的价值在于：")
print("以后你回头看结果时，知道自己改了什么，而不是只记得“当时这个模型好像更准”。")


# ============================================================
# 6. TODO(human) 练习
# ============================================================

print("\n" + "=" * 60)
print("6. TODO(human) 练习")
print("=" * 60)
print("TODO(human): 把分类阈值改成 0.3，再重新观察 precision / recall / F1 的变化。")
print("TODO(human): 给 experiment_log 多加一条实验记录，例如“标准化后再训练”的版本，并写下 valid 指标。")
