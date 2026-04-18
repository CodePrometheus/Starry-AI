"""
机器学习项目实战 (14) - 端到端迷你项目
=========================================
这一章把前面学过的内容串起来，完整走一遍：

1. 明确任务和目标
2. 准备并切分数据
3. 做预处理和特征工程
4. 训练一个小型 MLP 分类器
5. 在验证集和测试集上评估
6. 做最基础的误差分析
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


np.random.seed(42)


print("=" * 60)
print("1. 项目目标：预测二手房是否会在 14 天内成交")
print("=" * 60)

# 这是一个玩具版但足够真实的小项目。
# 每条样本表示一套正在挂牌的二手房。
#
# 目标字段 sold_within_14_days：
#   1 = 两周内成交
#   0 = 没有在两周内成交
#
# 我们只使用挂牌时就能知道的信息：
#   - area: 面积
#   - total_price: 挂牌总价（万元）
#   - distance_to_subway: 距离地铁（公里）
#   - building_age: 房龄（年）
#   - school_score: 学区吸引力（1~10）
#   - renovation: 装修情况（精装/简装/毛坯）
#   - district: 片区（核心区/次核心区/郊区）


raw_samples = [
    {"id": "H01", "area": 58, "total_price": 430, "distance_to_subway": 0.4, "building_age": 6, "school_score": 8, "renovation": "精装", "district": "核心区", "sold_within_14_days": 1},
    {"id": "H02", "area": 72, "total_price": 520, "distance_to_subway": 0.7, "building_age": 10, "school_score": 7, "renovation": "精装", "district": "次核心区", "sold_within_14_days": 1},
    {"id": "H03", "area": 96, "total_price": 780, "distance_to_subway": 1.6, "building_age": 18, "school_score": 6, "renovation": "简装", "district": "核心区", "sold_within_14_days": 0},
    {"id": "H04", "area": 82, "total_price": 560, "distance_to_subway": 0.9, "building_age": 15, "school_score": 8, "renovation": "简装", "district": "次核心区", "sold_within_14_days": 1},
    {"id": "H05", "area": 110, "total_price": 930, "distance_to_subway": 1.8, "building_age": 12, "school_score": 9, "renovation": "精装", "district": "核心区", "sold_within_14_days": 0},
    {"id": "H06", "area": 67, "total_price": 390, "distance_to_subway": 1.2, "building_age": 20, "school_score": 5, "renovation": "毛坯", "district": "郊区", "sold_within_14_days": 0},
    {"id": "H07", "area": 75, "total_price": 468, "distance_to_subway": 0.5, "building_age": 7, "school_score": 7, "renovation": "精装", "district": "次核心区", "sold_within_14_days": 1},
    {"id": "H08", "area": 88, "total_price": 610, "distance_to_subway": 1.3, "building_age": 11, "school_score": 6, "renovation": "简装", "district": "次核心区", "sold_within_14_days": 0},
    {"id": "H09", "area": 54, "total_price": 320, "distance_to_subway": 0.6, "building_age": 22, "school_score": 6, "renovation": "简装", "district": "郊区", "sold_within_14_days": 1},
    {"id": "H10", "area": 63, "total_price": 405, "distance_to_subway": 1.5, "building_age": 16, "school_score": 5, "renovation": "毛坯", "district": "次核心区", "sold_within_14_days": 0},
    {"id": "H11", "area": 92, "total_price": 680, "distance_to_subway": 0.8, "building_age": 9, "school_score": 8, "renovation": "精装", "district": "核心区", "sold_within_14_days": 1},
    {"id": "H12", "area": 118, "total_price": 990, "distance_to_subway": 2.0, "building_age": 5, "school_score": 9, "renovation": "精装", "district": "核心区", "sold_within_14_days": 0},
    {"id": "H13", "area": 70, "total_price": 438, "distance_to_subway": 0.9, "building_age": 8, "school_score": 7, "renovation": "精装", "district": "郊区", "sold_within_14_days": 1},
    {"id": "H14", "area": 84, "total_price": 535, "distance_to_subway": 1.1, "building_age": 13, "school_score": 6, "renovation": "简装", "district": "次核心区", "sold_within_14_days": 0},
    {"id": "H15", "area": 101, "total_price": 720, "distance_to_subway": 0.7, "building_age": 14, "school_score": 8, "renovation": "精装", "district": "核心区", "sold_within_14_days": 1},
    {"id": "H16", "area": 57, "total_price": 350, "distance_to_subway": 1.0, "building_age": 9, "school_score": 6, "renovation": "简装", "district": "郊区", "sold_within_14_days": 1},
    {"id": "H17", "area": 124, "total_price": 880, "distance_to_subway": 1.4, "building_age": 4, "school_score": 8, "renovation": "精装", "district": "次核心区", "sold_within_14_days": 0},
    {"id": "H18", "area": 69, "total_price": 410, "distance_to_subway": 0.6, "building_age": 17, "school_score": 5, "renovation": "简装", "district": "核心区", "sold_within_14_days": 1},
    {"id": "H19", "area": 80, "total_price": 490, "distance_to_subway": 1.7, "building_age": 19, "school_score": 4, "renovation": "毛坯", "district": "郊区", "sold_within_14_days": 0},
    {"id": "H20", "area": 98, "total_price": 705, "distance_to_subway": 0.5, "building_age": 6, "school_score": 9, "renovation": "精装", "district": "次核心区", "sold_within_14_days": 1},
    {"id": "H21", "area": 86, "total_price": 650, "distance_to_subway": 1.9, "building_age": 7, "school_score": 9, "renovation": "精装", "district": "核心区", "sold_within_14_days": 0},
    {"id": "H22", "area": 62, "total_price": 360, "distance_to_subway": 0.8, "building_age": 21, "school_score": 5, "renovation": "简装", "district": "郊区", "sold_within_14_days": 1},
    {"id": "H23", "area": 106, "total_price": 760, "distance_to_subway": 1.2, "building_age": 8, "school_score": 7, "renovation": "精装", "district": "次核心区", "sold_within_14_days": 0},
    {"id": "H24", "area": 74, "total_price": 458, "distance_to_subway": 0.7, "building_age": 12, "school_score": 6, "renovation": "精装", "district": "郊区", "sold_within_14_days": 1},
]

print(f"样本总数 = {len(raw_samples)}")
print("前 5 条样本：")
for sample in raw_samples[:5]:
    print(sample)


print("\n" + "=" * 60)
print("2. 切分数据集")
print("=" * 60)

# 为了让脚本每次运行结果一致，这里直接手动切分。
train_rows = raw_samples[:16]
valid_rows = raw_samples[16:20]
test_rows = raw_samples[20:]

print(f"训练集大小 = {len(train_rows)}")
print(f"验证集大小 = {len(valid_rows)}")
print(f"测试集大小 = {len(test_rows)}")


print("\n" + "=" * 60)
print("3. 预处理与特征工程")
print("=" * 60)


def enrich_sample(row: dict) -> dict:
    """从原始字段构造两个更有业务意义的特征。"""
    enriched = dict(row)
    enriched["price_per_sqm"] = row["total_price"] / row["area"]
    enriched["near_subway"] = 1 if row["distance_to_subway"] <= 0.8 else 0
    return enriched


train_rows = [enrich_sample(row) for row in train_rows]
valid_rows = [enrich_sample(row) for row in valid_rows]
test_rows = [enrich_sample(row) for row in test_rows]

print("新增特征：")
print("  price_per_sqm = total_price / area")
print("  near_subway   = 1(距离地铁 <= 0.8km) / 0")
print(f"训练集第一条样本新增后 = {train_rows[0]}")

numeric_features = [
    "area",
    "total_price",
    "distance_to_subway",
    "building_age",
    "school_score",
    "price_per_sqm",
    "near_subway",
]
categorical_features = ["district", "renovation"]


def fit_categories(rows: list[dict], feature_name: str) -> list[str]:
    """只在训练集上确定类别词表。"""
    return sorted({row[feature_name] for row in rows})


district_vocab = fit_categories(train_rows, "district")
renovation_vocab = fit_categories(train_rows, "renovation")

print(f"\ndistrict 词表 = {district_vocab}")
print(f"renovation 词表 = {renovation_vocab}")


def rows_to_numeric_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    """把多条样本的数值字段抽成矩阵。"""
    return np.array([[row[name] for name in feature_names] for row in rows], dtype=float)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """在训练集上计算标准化参数。"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return mean, std_safe


def transform_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """用训练集的统计量做标准化。"""
    return (X - mean) / std


def one_hot(value: str, vocab: list[str]) -> np.ndarray:
    """把单个类别值编码成 One-Hot。"""
    return np.array([1.0 if value == token else 0.0 for token in vocab], dtype=float)


def build_feature_matrix(rows: list[dict], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """把数值特征和类别特征拼成最终输入矩阵。"""
    X_numeric = rows_to_numeric_matrix(rows, numeric_features)
    X_numeric_scaled = transform_standardize(X_numeric, mean, std)

    cat_vectors = []
    for row in rows:
        district_vector = one_hot(row["district"], district_vocab)
        renovation_vector = one_hot(row["renovation"], renovation_vocab)
        cat_vectors.append(np.concatenate([district_vector, renovation_vector]))

    X_categorical = np.array(cat_vectors, dtype=float)
    return np.concatenate([X_numeric_scaled, X_categorical], axis=1)


X_train_numeric = rows_to_numeric_matrix(train_rows, numeric_features)
train_mean, train_std = fit_standardizer(X_train_numeric)

X_train = build_feature_matrix(train_rows, train_mean, train_std)
X_valid = build_feature_matrix(valid_rows, train_mean, train_std)
X_test = build_feature_matrix(test_rows, train_mean, train_std)

y_train = np.array([row["sold_within_14_days"] for row in train_rows], dtype=float).reshape(-1, 1)
y_valid = np.array([row["sold_within_14_days"] for row in valid_rows], dtype=float).reshape(-1, 1)
y_test = np.array([row["sold_within_14_days"] for row in test_rows], dtype=float).reshape(-1, 1)

print(f"\n最终训练特征矩阵形状 = {X_train.shape}")
print(f"最终验证特征矩阵形状 = {X_valid.shape}")
print(f"最终测试特征矩阵形状 = {X_test.shape}")
print("训练集第一条样本的最终特征向量：")
print(np.round(X_train[0], 3))


print("\n" + "=" * 60)
print("4. 训练一个小型 MLP")
print("=" * 60)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活。"""
    return np.maximum(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 激活。"""
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TinyMLP:
    """一个最小可运行的二分类 MLP。"""

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


def init_model(input_dim: int, hidden_dim: int) -> TinyMLP:
    """He 初始化，更适合 ReLU 隐藏层。"""
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros((1, hidden_dim), dtype=float)
    W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros((1, 1), dtype=float)
    return TinyMLP(W1=W1, b1=b1, W2=W2, b2=b2)


def forward(model: TinyMLP, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """前向传播。"""
    Z1 = X @ model.W1 + model.b1
    A1 = relu(Z1)
    Z2 = A1 @ model.W2 + model.b2
    probs = sigmoid(Z2)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "probs": probs}
    return probs, cache


def binary_cross_entropy(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """二分类交叉熵。"""
    eps = 1e-8
    return -np.mean(y_true * np.log(y_prob + eps) + (1 - y_true) * np.log(1 - y_prob + eps))


def compute_loss(model: TinyMLP, X: np.ndarray, y: np.ndarray, l2_lambda: float) -> float:
    """交叉熵 + L2 正则。"""
    probs, _ = forward(model, X)
    data_loss = binary_cross_entropy(probs, y)
    l2_loss = 0.5 * l2_lambda * (np.sum(model.W1 ** 2) + np.sum(model.W2 ** 2))
    return float(data_loss + l2_loss)


def backward(
    model: TinyMLP,
    cache: dict[str, np.ndarray],
    y_true: np.ndarray,
    l2_lambda: float,
) -> dict[str, np.ndarray]:
    """反向传播求梯度。"""
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    probs = cache["probs"]

    n = X.shape[0]
    dZ2 = (probs - y_true) / n
    dW2 = A1.T @ dZ2 + l2_lambda * model.W2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ model.W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1 + l2_lambda * model.W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update(model: TinyMLP, grads: dict[str, np.ndarray], learning_rate: float) -> None:
    """梯度下降更新参数。"""
    model.W1 -= learning_rate * grads["dW1"]
    model.b1 -= learning_rate * grads["db1"]
    model.W2 -= learning_rate * grads["dW2"]
    model.b2 -= learning_rate * grads["db2"]


def predict(model: TinyMLP, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """输出类别概率和最终类别。"""
    probs, _ = forward(model, X)
    labels = (probs >= 0.5).astype(float)
    return probs, labels


model = init_model(input_dim=X_train.shape[1], hidden_dim=8)
learning_rate = 0.08
l2_lambda = 0.001
epochs = 600

best_model = copy.deepcopy(model)
best_valid_loss = float("inf")

for epoch in range(1, epochs + 1):
    train_probs, train_cache = forward(model, X_train)
    train_loss = binary_cross_entropy(train_probs, y_train)

    grads = backward(model, train_cache, y_train, l2_lambda=l2_lambda)
    update(model, grads, learning_rate=learning_rate)

    valid_loss = compute_loss(model, X_valid, y_valid, l2_lambda=l2_lambda)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_model = copy.deepcopy(model)

    if epoch in {1, 50, 100, 200, 400, 600}:
        print(
            f"epoch={epoch:>3d} | "
            f"train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f}"
        )

model = best_model
print(f"\n最佳验证集 loss = {best_valid_loss:.4f}")


print("\n" + "=" * 60)
print("5. 评估：准确率、精确率、召回率、混淆矩阵")
print("=" * 60)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算二分类常见指标。"""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    tp = int(np.sum((y_true_flat == 1) & (y_pred_flat == 1)))
    tn = int(np.sum((y_true_flat == 0) & (y_pred_flat == 0)))
    fp = int(np.sum((y_true_flat == 0) & (y_pred_flat == 1)))
    fn = int(np.sum((y_true_flat == 1) & (y_pred_flat == 0)))

    accuracy = (tp + tn) / len(y_true_flat)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def print_split_metrics(split_name: str, X: np.ndarray, y: np.ndarray) -> None:
    """打印一个数据集切分上的评估结果。"""
    probs, preds = predict(model, X)
    metrics = classification_report(y, preds)
    print(f"\n{split_name}")
    print(f"  accuracy  = {metrics['accuracy']:.2%}")
    print(f"  precision = {metrics['precision']:.2%}")
    print(f"  recall    = {metrics['recall']:.2%}")
    print(
        f"  confusion = [[TN={metrics['tn']}, FP={metrics['fp']}], "
        f"[FN={metrics['fn']}, TP={metrics['tp']}]]"
    )
    print(f"  概率示例 = {np.round(probs.reshape(-1), 4)}")


print_split_metrics("训练集", X_train, y_train)
print_split_metrics("验证集", X_valid, y_valid)
print_split_metrics("测试集", X_test, y_test)


print("\n" + "=" * 60)
print("6. 误差分析：模型在哪些房子上容易看错？")
print("=" * 60)

test_probs, test_preds = predict(model, X_test)


def sample_reason(row: dict) -> str:
    """给每条样本生成一个简短的人类可读解释。"""
    reasons = []
    if row["near_subway"] == 1:
        reasons.append("离地铁近")
    if row["price_per_sqm"] > 8.0:
        reasons.append("单价偏高")
    if row["building_age"] >= 18:
        reasons.append("房龄偏老")
    if row["school_score"] >= 8:
        reasons.append("学区分高")
    if row["renovation"] == "毛坯":
        reasons.append("装修弱")
    if not reasons:
        reasons.append("特征不算特别突出")
    return "、".join(reasons)


mistakes = []
for row, prob, pred, label in zip(test_rows, test_probs.reshape(-1), test_preds.reshape(-1), y_test.reshape(-1)):
    if pred != label:
        mistakes.append((row, prob, pred, label))

if mistakes:
    print("测试集中的误分类样本：")
    for row, prob, pred, label in mistakes:
        print(
            f"  {row['id']}: 预测={int(pred)}(概率={prob:.4f}), 真实={int(label)}, "
            f"原因线索={sample_reason(row)}"
        )
else:
    print("测试集没有误分类样本，改看最不确定的样本：")
    uncertainty = np.abs(test_probs.reshape(-1) - 0.5)
    sorted_indices = np.argsort(uncertainty)
    for idx in sorted_indices[:2]:
        row = test_rows[idx]
        prob = test_probs[idx, 0]
        pred = test_preds[idx, 0]
        label = y_test[idx, 0]
        print(
            f"  {row['id']}: 预测={int(pred)}(概率={prob:.4f}), 真实={int(label)}, "
            f"原因线索={sample_reason(row)}"
        )

print("\n从项目角度看，这一步的价值在于：")
print("  不是只盯着一个 accuracy，而是回到具体样本，判断模型错在哪。")
print("  如果总是把“地铁远但学区好”的房子看错，下一步就该补这类样本。")


print("\n" + "=" * 60)
print("7. 练习题")
print("=" * 60)
print("TODO(human): 把 district 再细化成更多片区，或者增加“楼层/是否满五唯一”等字段，")
print("重新训练后比较验证集与测试集表现，看特征增加是否真的带来收益。")
