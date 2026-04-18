"""
机器学习基础 (7) - 决策树与集成
====================================
这一章解决的问题是：

1. 决策树为什么像一串 if / else 规则？
2. 一个切分点好不好，怎么用 impurity / information gain 判断？
3. 为什么把多棵树投票，常常比单棵树更稳？

本脚本重点演示：
- 决策树切分直觉
- Gini impurity 与 entropy / information gain
- 从零构造一个浅层分类树
- bagging / random-forest 风格的多数投票示例
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 1. 准备一个户外活动分类数据集
# ============================================================
# 场景：根据当天气温和降雨量，判断“晨跑团今天适不适合组织户外活动”。
#
# join = 1 表示适合组织
# join = 0 表示不适合组织
#
# 直觉上：
#   - 雨太大通常不适合
#   - 太冷或太热也不适合
#   - 温度适中且雨小，最适合

print("=" * 60)
print("1. 分类数据")
print("=" * 60)

samples = [
    {"temperature_c": 16, "rainfall_mm": 0.0, "join": 0},
    {"temperature_c": 18, "rainfall_mm": 0.2, "join": 0},
    {"temperature_c": 20, "rainfall_mm": 0.0, "join": 1},
    {"temperature_c": 22, "rainfall_mm": 1.5, "join": 0},
    {"temperature_c": 24, "rainfall_mm": 0.3, "join": 1},
    {"temperature_c": 26, "rainfall_mm": 0.1, "join": 1},
    {"temperature_c": 28, "rainfall_mm": 1.8, "join": 0},
    {"temperature_c": 30, "rainfall_mm": 0.4, "join": 1},
    {"temperature_c": 32, "rainfall_mm": 0.2, "join": 1},
    {"temperature_c": 34, "rainfall_mm": 2.2, "join": 0},
    {"temperature_c": 36, "rainfall_mm": 0.6, "join": 0},
    {"temperature_c": 38, "rainfall_mm": 0.1, "join": 0},
]

for i, sample in enumerate(samples, start=1):
    print(
        f"样本{i:02d}: temperature = {sample['temperature_c']:>2d}°C, "
        f"rainfall = {sample['rainfall_mm']:.1f} mm, join = {sample['join']}"
    )

feature_names = ["temperature_c", "rainfall_mm"]
X = np.array([[sample[name] for name in feature_names] for sample in samples], dtype=float)
y = np.array([sample["join"] for sample in samples], dtype=int)


# ============================================================
# 2. 一个切分点为什么好：Gini impurity 和 entropy
# ============================================================
# 决策树的核心就是不断问：
#   “如果我在某个特征、某个阈值上切一刀，左右两边会不会更纯？”
#
# 这里的“更纯”就是 impurity 更低。
#
# Gini impurity：
#   Gini = 1 - p1^2 - p0^2
#
# entropy：
#   entropy = -p1*log2(p1) - p0*log2(p0)
#
# 二者的共同目标都是：
#   让分裂后的每个子节点尽量只剩一类样本。

print("\n" + "=" * 60)
print("2. impurity 和 information gain")
print("=" * 60)


def gini_impurity(labels: np.ndarray) -> float:
    """计算二分类标签的 Gini impurity。"""
    if len(labels) == 0:
        return 0.0
    p1 = np.mean(labels)
    p0 = 1.0 - p1
    return float(1.0 - p1 ** 2 - p0 ** 2)


def entropy(labels: np.ndarray) -> float:
    """计算二分类标签的熵。"""
    if len(labels) == 0:
        return 0.0

    p1 = np.mean(labels)
    p0 = 1.0 - p1
    value = 0.0
    for probability in [p0, p1]:
        if probability > 0:
            value -= probability * np.log2(probability)
    return float(value)


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    """相邻取值的中点，常用于连续特征候选切分点。"""
    unique_values = np.unique(values)
    return (unique_values[:-1] + unique_values[1:]) / 2.0


def evaluate_split(feature_index: int, threshold: float) -> dict[str, float]:
    """评估在某个特征和阈值上的切分质量。"""
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    left_labels = y[left_mask]
    right_labels = y[right_mask]

    base_gini = gini_impurity(y)
    weighted_gini = (
        len(left_labels) / len(y) * gini_impurity(left_labels)
        + len(right_labels) / len(y) * gini_impurity(right_labels)
    )
    gini_gain = base_gini - weighted_gini

    base_entropy = entropy(y)
    weighted_entropy = (
        len(left_labels) / len(y) * entropy(left_labels)
        + len(right_labels) / len(y) * entropy(right_labels)
    )
    info_gain = base_entropy - weighted_entropy

    return {
        "feature_index": feature_index,
        "threshold": float(threshold),
        "left_size": int(left_mask.sum()),
        "right_size": int(right_mask.sum()),
        "weighted_gini": float(weighted_gini),
        "gini_gain": float(gini_gain),
        "info_gain": float(info_gain),
    }


root_gini = gini_impurity(y)
root_entropy = entropy(y)
print(f"根节点上共有 {len(y)} 个样本，其中正类比例 = {np.mean(y):.4f}")
print(f"根节点 Gini impurity = {root_gini:.4f}")
print(f"根节点 entropy       = {root_entropy:.4f}")

all_split_results: list[dict[str, float]] = []
for feature_index, feature_name in enumerate(feature_names):
    print(f"\n候选特征: {feature_name}")
    for threshold in candidate_thresholds(X[:, feature_index]):
        result = evaluate_split(feature_index, float(threshold))
        all_split_results.append(result)
        print(
            f"  threshold = {threshold:>5.2f} -> "
            f"weighted_gini = {result['weighted_gini']:.4f}, "
            f"gini_gain = {result['gini_gain']:.4f}, "
            f"info_gain = {result['info_gain']:.4f}"
        )

best_root_split = max(all_split_results, key=lambda item: item["gini_gain"])
best_feature_name = feature_names[int(best_root_split["feature_index"])]
print("\n最佳根节点切分（按 Gini gain 选）:")
print(
    f"  {best_feature_name} <= {best_root_split['threshold']:.2f}, "
    f"gini_gain = {best_root_split['gini_gain']:.4f}, "
    f"info_gain = {best_root_split['info_gain']:.4f}"
)
print("→ 这里最先学到的规则是：如果雨量已经偏大，样本会很快变纯。")


# ============================================================
# 3. 从零构造一棵浅层决策树
# ============================================================
# 决策树就是递归地重复上面的过程：
#   - 先找当前节点最好的切分点
#   - 把数据分到左子树 / 右子树
#   - 对子树继续切分
#   - 如果已经很纯、或者达到最大深度，就停下来做叶子节点

print("\n" + "=" * 60)
print("3. 从零构造浅层树")
print("=" * 60)


def majority_label(labels: np.ndarray) -> int:
    """叶子节点预测该节点里的多数类。"""
    return int(np.mean(labels) >= 0.5)


def find_best_split(
    X_node: np.ndarray,
    y_node: np.ndarray,
    candidate_feature_indices: np.ndarray | None = None,
) -> dict[str, object] | None:
    """在当前节点里找最优切分。"""
    if len(np.unique(y_node)) == 1:
        return None

    if candidate_feature_indices is None:
        candidate_feature_indices = np.arange(X_node.shape[1])

    best_result: dict[str, object] | None = None
    base_gini = gini_impurity(y_node)

    for feature_index in candidate_feature_indices:
        thresholds = candidate_thresholds(X_node[:, int(feature_index)])
        for threshold in thresholds:
            left_mask = X_node[:, int(feature_index)] <= threshold
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            left_labels = y_node[left_mask]
            right_labels = y_node[right_mask]
            weighted_gini = (
                len(left_labels) / len(y_node) * gini_impurity(left_labels)
                + len(right_labels) / len(y_node) * gini_impurity(right_labels)
            )
            gain = base_gini - weighted_gini

            if best_result is None or gain > best_result["gain"]:
                best_result = {
                    "feature_index": int(feature_index),
                    "threshold": float(threshold),
                    "gain": float(gain),
                    "left_mask": left_mask,
                }

    return best_result


def build_tree(
    X_node: np.ndarray,
    y_node: np.ndarray,
    depth: int,
    max_depth: int,
    rng: np.random.Generator | None = None,
    max_features: int | None = None,
) -> dict[str, object]:
    """递归构造一个分类树。"""
    if depth >= max_depth or len(np.unique(y_node)) == 1:
        return {
            "type": "leaf",
            "label": majority_label(y_node),
            "size": int(len(y_node)),
        }

    if max_features is None:
        candidate_feature_indices = np.arange(X_node.shape[1])
    else:
        assert rng is not None
        candidate_feature_indices = rng.choice(
            X_node.shape[1],
            size=max_features,
            replace=False,
        )

    best_split = find_best_split(X_node, y_node, candidate_feature_indices)
    if best_split is None or best_split["gain"] <= 0:
        return {
            "type": "leaf",
            "label": majority_label(y_node),
            "size": int(len(y_node)),
        }

    left_mask = best_split["left_mask"]
    return {
        "type": "node",
        "feature_index": int(best_split["feature_index"]),
        "threshold": float(best_split["threshold"]),
        "gain": float(best_split["gain"]),
        "left": build_tree(
            X_node[left_mask],
            y_node[left_mask],
            depth=depth + 1,
            max_depth=max_depth,
            rng=rng,
            max_features=max_features,
        ),
        "right": build_tree(
            X_node[~left_mask],
            y_node[~left_mask],
            depth=depth + 1,
            max_depth=max_depth,
            rng=rng,
            max_features=max_features,
        ),
    }


def predict_one(tree: dict[str, object], x: np.ndarray) -> int:
    """用一棵树对单条样本做预测。"""
    if tree["type"] == "leaf":
        return int(tree["label"])

    feature_index = int(tree["feature_index"])
    threshold = float(tree["threshold"])
    if x[feature_index] <= threshold:
        return predict_one(tree["left"], x)
    return predict_one(tree["right"], x)


def describe_tree(tree: dict[str, object], depth: int = 0) -> None:
    """把树打印成可读规则。"""
    indent = "  " * depth
    if tree["type"] == "leaf":
        print(f"{indent}叶子: predict = {tree['label']}, size = {tree['size']}")
        return

    feature_name = feature_names[int(tree["feature_index"])]
    print(
        f"{indent}如果 {feature_name} <= {tree['threshold']:.2f} "
        f"(gain = {tree['gain']:.4f})"
    )
    describe_tree(tree["left"], depth + 1)
    print(f"{indent}否则")
    describe_tree(tree["right"], depth + 1)


single_tree = build_tree(X, y, depth=0, max_depth=2)
print("构造出的深度 2 决策树规则：")
describe_tree(single_tree)

single_tree_predictions = np.array([predict_one(single_tree, row) for row in X], dtype=int)
single_tree_accuracy = np.mean(single_tree_predictions == y)
print(f"\n单棵树在训练集上的预测 = {single_tree_predictions}")
print(f"单棵树训练集准确率 = {single_tree_accuracy:.4f}")


# ============================================================
# 4. 集成 intuition：多棵树投票为什么更稳
# ============================================================
# 单棵树有个典型问题：高方差。
# 也就是说，训练数据稍微变一点，切分规则就可能明显变化。
#
# bagging 的思路是：
#   1. 对训练集做 bootstrap 重采样
#   2. 每次训练一棵树
#   3. 最后让这些树多数投票
#
# random forest 还会进一步在每次切分时随机挑一部分特征，
# 让不同树之间差异更大。

print("\n" + "=" * 60)
print("4. 集成 intuition：多数投票")
print("=" * 60)

rng = np.random.default_rng(42)
forest: list[dict[str, object]] = []
n_trees = 7

for tree_idx in range(n_trees):
    bootstrap_indices = rng.choice(len(X), size=len(X), replace=True)
    tree = build_tree(
        X[bootstrap_indices],
        y[bootstrap_indices],
        depth=0,
        max_depth=2,
        rng=rng,
        max_features=1,
    )
    forest.append(tree)

    if tree["type"] == "node":
        root_feature = feature_names[int(tree["feature_index"])]
        print(
            f"树{tree_idx + 1}: 根节点用 {root_feature} <= {tree['threshold']:.2f}"
        )
    else:
        print(f"树{tree_idx + 1}: 直接成为叶子节点，predict = {tree['label']}")


def forest_predict(forest: list[dict[str, object]], x: np.ndarray) -> tuple[int, np.ndarray]:
    """让多棵树投票。"""
    votes = np.array([predict_one(tree, x) for tree in forest], dtype=int)
    final_prediction = int(np.mean(votes) >= 0.5)
    return final_prediction, votes


query_days = np.array(
    [
        [21.0, 0.2],  # 温度刚刚合适、雨也不大
        [27.0, 1.0],  # 温度不错，但已经有明显降雨
        [35.0, 0.3],  # 雨不大，但已经偏热
        [29.0, 1.9],  # 温度不错，但雨太大
    ],
    dtype=float,
)

print("\n用森林看几条新样本：")
for query in query_days:
    final_prediction, votes = forest_predict(forest, query)
    print(
        f"  query = [temperature={query[0]:.1f}, rainfall={query[1]:.1f}] -> "
        f"votes = {votes}, final = {final_prediction}"
    )

print("\n→ 你会看到：不同树不一定完全一致，但多数投票通常更稳。")


# ============================================================
# 5. 动手练习
# ============================================================

print("\n" + "=" * 60)
print("5. 动手练习")
print("=" * 60)
print("TODO(human): 把树的 max_depth 从 2 改成 3，观察单棵树训练集准确率会不会继续上升。")
print("TODO(human): 把 find_best_split 里的 Gini 换成 entropy，再看根节点切分是否变化。")
