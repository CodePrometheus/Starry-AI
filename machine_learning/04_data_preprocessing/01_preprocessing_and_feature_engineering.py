"""
机器学习基础 (4) - 数据预处理与特征工程
========================================
这一章解决两个问题：

1. 原始数据为什么不能直接喂给模型？
2. 如何把原始字段处理成更适合模型学习的特征？

本脚本重点演示：
- 训练集 / 验证集 / 测试集切分
- 标准化（standardization）
- 类别特征的 One-Hot 编码
- 手工构造特征（feature engineering）
- 为什么预处理只能在训练集上 fit，不能偷看验证集和测试集
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 1. 准备一个小型表格数据集
# ============================================================
# 我们用一个简化版房价数据来学习预处理。
# 每一条样本包含：
#   - area: 房屋面积（平方米）
#   - rooms: 房间数
#   - distance_to_subway: 距离地铁站距离（公里）
#   - age: 房龄（年）
#   - district: 区域（类别特征）
#   - price: 房价（目标值，暂时只用于观察，不参与预处理）
#
# 特别注意：
#   最后两条测试样本明显更大更贵，这是故意设计的。
#   这样可以更直观地看到：如果你偷看测试集再做标准化，会把未来信息泄漏给训练过程。

print("=" * 60)
print("1. 原始数据")
print("=" * 60)

samples = [
    {"area": 52, "rooms": 2, "distance_to_subway": 1.3, "age": 18, "district": "A", "price": 320},
    {"area": 60, "rooms": 2, "distance_to_subway": 0.9, "age": 12, "district": "A", "price": 380},
    {"area": 74, "rooms": 3, "distance_to_subway": 1.8, "age": 20, "district": "B", "price": 410},
    {"area": 81, "rooms": 3, "distance_to_subway": 0.7, "age": 8, "district": "B", "price": 520},
    {"area": 68, "rooms": 2, "distance_to_subway": 2.1, "age": 25, "district": "C", "price": 360},
    {"area": 91, "rooms": 4, "distance_to_subway": 1.1, "age": 7, "district": "A", "price": 610},
    {"area": 88, "rooms": 3, "distance_to_subway": 0.6, "age": 6, "district": "C", "price": 640},
    {"area": 77, "rooms": 3, "distance_to_subway": 1.5, "age": 14, "district": "B", "price": 470},
    {"area": 95, "rooms": 4, "distance_to_subway": 0.8, "age": 5, "district": "A", "price": 690},
    {"area": 104, "rooms": 4, "distance_to_subway": 1.0, "age": 9, "district": "C", "price": 720},
    {"area": 185, "rooms": 6, "distance_to_subway": 0.5, "age": 3, "district": "B", "price": 1350},
    {"area": 210, "rooms": 7, "distance_to_subway": 0.4, "age": 2, "district": "A", "price": 1600},
]

for i, sample in enumerate(samples, start=1):
    print(f"样本{i:02d}: {sample}")


# ============================================================
# 2. 训练集 / 验证集 / 测试集切分
# ============================================================
# 切分的核心目的：
#   - 训练集：拿来学习参数
#   - 验证集：拿来调超参数、做模型选择
#   - 测试集：最后一次客观评估，不参与训练和调参
#
# 这里为了演示稳定，我们直接手动切分：
#   0~7   -> train
#   8~9   -> valid
#   10~11 -> test
#
# 在真实项目里，常见做法是随机打乱后再切分，但逻辑完全一致。

print("\n" + "=" * 60)
print("2. 数据切分")
print("=" * 60)

train_samples = samples[:8]
valid_samples = samples[8:10]
test_samples = samples[10:]

print(f"训练集大小: {len(train_samples)}")
print(f"验证集大小: {len(valid_samples)}")
print(f"测试集大小: {len(test_samples)}")


# ============================================================
# 3. 特征工程：从原始字段构造更有用的特征
# ============================================================
# 原始字段不一定就是最适合模型学习的表达。
# 所以我们经常会手工构造一些新特征。
#
# 下面构造两个典型特征：
#   (1) area_per_room = area / rooms
#       直觉：平均每个房间有多大，比单独看面积或房间数更细
#   (2) near_subway = 1 if distance_to_subway <= 1.0 else 0
#       直觉：是否靠近地铁，变成一个二值特征

print("\n" + "=" * 60)
print("3. 特征工程")
print("=" * 60)


def add_engineered_features(sample: dict) -> dict:
    """基于原始字段构造新特征。"""
    area_per_room = sample["area"] / sample["rooms"]
    near_subway = 1 if sample["distance_to_subway"] <= 1.0 else 0

    enriched = dict(sample)
    enriched["area_per_room"] = area_per_room
    enriched["near_subway"] = near_subway
    return enriched


train_samples = [add_engineered_features(sample) for sample in train_samples]
valid_samples = [add_engineered_features(sample) for sample in valid_samples]
test_samples = [add_engineered_features(sample) for sample in test_samples]

print("手工构造的两个新特征：")
print("  area_per_room = area / rooms")
print("  near_subway   = 1(近地铁) / 0(不近地铁)")
print("\n特征工程后的训练集样本:")
for sample in train_samples:
    print(sample)
print("\n特征工程后的验证集样本:")
for sample in valid_samples:
    print(sample)
print("\n特征工程后的测试集样本:")
for sample in test_samples:
    print(sample)


# ============================================================
# 4. 类别特征处理：One-Hot 编码
# ============================================================
# 机器学习模型一般不能直接处理 "A / B / C" 这种字符串。
# 所以我们需要把类别特征转换成数值。
#
# 最常见的基本做法之一就是 One-Hot 编码：
#   district = "A" -> [1, 0, 0]
#   district = "B" -> [0, 1, 0]
#   district = "C" -> [0, 0, 1]
#
# 注意：
#   类别字典也应该只在训练集上确定。
#   如果测试集中才出现新类别，真实项目里要单独设计 OOV / unknown 策略。

print("\n" + "=" * 60)
print("4. One-Hot 编码")
print("=" * 60)


def fit_one_hot_categories(train_rows: list[dict], key: str) -> list[str]:
    """只在训练集上收集类别。"""
    categories = sorted({row[key] for row in train_rows})
    return categories


def transform_one_hot(value: str, categories: list[str]) -> np.ndarray:
    """把单个类别值转成 One-Hot 向量。"""
    return np.array([1 if value == category else 0 for category in categories], dtype=float)


district_categories = fit_one_hot_categories(train_samples, "district")
print(f"训练集里出现的区域类别: {district_categories}")
print(f"district='B' 的 One-Hot 编码: {transform_one_hot('B', district_categories)}")


# ============================================================
# 5. 数值特征处理：标准化
# ============================================================
# 不同特征的量纲经常差很多。
# 例如：
#   area 是几十到几百
#   distance_to_subway 是 0.x 到 2.x
#   age 是几年到几十年
#
# 如果不处理，数值范围大的特征可能在优化时更“显眼”。
# 所以常见做法是标准化：
#
#   z = (x - μ) / σ
#
# 含义：
#   - 减去均值 μ ：让特征以 0 为中心
#   - 除以标准差 σ：让特征尺度变得可比
#
# 非常重要：
#   μ 和 σ 必须只在训练集上计算。
#   绝不能把验证集、测试集一起拿来算，这就叫数据泄漏。

print("\n" + "=" * 60)
print("5. 标准化")
print("=" * 60)

numeric_feature_names = [
    "area",
    "rooms",
    "distance_to_subway",
    "age",
    "area_per_room",
    "near_subway",
]


def rows_to_numeric_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    """把多个样本里的数值特征抽成矩阵。"""
    return np.array([[row[name] for name in feature_names] for row in rows], dtype=float)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    在训练集上计算标准化所需的均值和标准差。

    这里的 X 是一个二维矩阵：
      - 每一行是一个样本
      - 每一列是一个特征

    所以：
      X.mean(axis=0) 的意思是“对每一列分别求平均数”
      X.std(axis=0)  的意思是“对每一列分别求标准差”

    例如：
      如果 area 这一列是 [52, 60, 74, 81, 68, 91, 88, 77]
      那么它的均值就是：
        (52 + 60 + 74 + 81 + 68 + 91 + 88 + 77) / 8 = 73.875

      这就是训练集数值特征均值向量里第一个数字的来源。

    这里给标准差加一个保护：
    如果某一列标准差为 0，说明这一列是常数列，直接把标准差设成 1，
    这样 transform 时不会除以 0。
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return mean, std_safe


def transform_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """根据给定的 mean / std 做标准化。"""
    return (X - mean) / std


X_train_numeric = rows_to_numeric_matrix(train_samples, numeric_feature_names)
X_valid_numeric = rows_to_numeric_matrix(valid_samples, numeric_feature_names)
X_test_numeric = rows_to_numeric_matrix(test_samples, numeric_feature_names)

print("训练集数值特征矩阵 X_train_numeric:")
print(np.round(X_train_numeric, 3))
print("\n验证集数值特征矩阵 X_valid_numeric:")
print(np.round(X_valid_numeric, 3))
print("\n测试集数值特征矩阵 X_test_numeric:")
print(np.round(X_test_numeric, 3))

print("\n按列计算均值和标准差的详细过程（只看训练集）:")
for col_idx, feature_name in enumerate(numeric_feature_names):
    column_values = X_train_numeric[:, col_idx]
    column_mean = column_values.mean()
    column_std = column_values.std()
    values_str = ", ".join([f"{value:.3f}" for value in column_values])
    sum_str = " + ".join([f"{value:.3f}" for value in column_values])
    print(f"\n特征 {feature_name}:")
    print(f"  这一列的值 = [{values_str}]")
    print(f"  均值 mean = ({sum_str}) / {len(column_values)} = {column_mean:.3f}")
    print(f"  标准差 std = {column_std:.3f}")
print("\n→ 所以 mean/std 不是对整张表只算一个数，而是每一列各算一个数。")

train_mean, train_std = fit_standardizer(X_train_numeric)

X_train_scaled = transform_standardize(X_train_numeric, train_mean, train_std)
X_valid_scaled = transform_standardize(X_valid_numeric, train_mean, train_std)
X_test_scaled = transform_standardize(X_test_numeric, train_mean, train_std)

print("训练集数值特征均值:")
print(np.round(train_mean, 3))
print("\n训练集数值特征标准差:")
print(np.round(train_std, 3))
print("\n标准化后的训练集前2条样本:")
print(np.round(X_train_scaled[:2], 3))


# ============================================================
# 6. 数据泄漏演示：错误做法 vs 正确做法
# ============================================================
# 错误做法：
#   拿“训练集 + 验证集 + 测试集”一起算均值和标准差
#
# 为什么错？
#   因为你在训练前偷看了未来数据分布。
#   这样预处理参数（均值、标准差、类别字典等）已经包含了测试集信息。
#
# 正确做法：
#   只在训练集上 fit
#   然后用训练集得到的参数去 transform 验证集和测试集

print("\n" + "=" * 60)
print("6. 数据泄漏演示")
print("=" * 60)

X_all_numeric = np.vstack([X_train_numeric, X_valid_numeric, X_test_numeric])
all_mean, all_std = fit_standardizer(X_all_numeric)

print("错误做法：把全部数据一起 fit 标准化器")
print(f"  全数据 mean[area]   = {all_mean[0]:.3f}")
print(f"  训练集 mean[area]   = {train_mean[0]:.3f}")
print(f"  全数据 std[area]    = {all_std[0]:.3f}")
print(f"  训练集 std[area]    = {train_std[0]:.3f}")

correct_test_area_scaled = X_test_scaled[:, 0]
wrong_test_area_scaled = transform_standardize(X_test_numeric, all_mean, all_std)[:, 0]

print("\n测试集 area 列的标准化结果对比：")
print(f"  正确做法（只用训练集 fit）: {np.round(correct_test_area_scaled, 3)}")
print(f"  错误做法（偷看全数据 fit）: {np.round(wrong_test_area_scaled, 3)}")
print("\n→ 你会看到数值明显不同，这说明预处理参数已经被测试集“污染”了。")


# ============================================================
# 7. 拼接最终特征矩阵
# ============================================================
# 完整预处理通常要把：
#   - 数值特征（标准化后）
#   - 类别特征（One-Hot 后）
# 拼接在一起
#
# 这里我们手动把最终特征矩阵做出来。

print("\n" + "=" * 60)
print("7. 最终特征矩阵")
print("=" * 60)


def build_final_feature_matrix(rows: list[dict], scaled_numeric: np.ndarray, categories: list[str]) -> np.ndarray:
    """
    把最终可喂给模型的特征矩阵拼出来。

    这里会做两件事：
    1. 从每条样本里取出 district，并把它变成 One-Hot 向量
    2. 把“标准化后的数值特征”与“One-Hot 类别特征”横向拼接

    参数说明：
    - rows:
        仍然保留字典形式的样本，目的是从里面读取 district 这个类别字段
    - scaled_numeric:
        已经标准化好的数值特征矩阵
        形状 = (样本数, 数值特征数)
    - categories:
        训练集上拟合出来的类别顺序，例如 ["A", "B", "C"]

    返回：
    - final_matrix:
        最终特征矩阵
        形状 = (样本数, 数值特征数 + 类别特征数)
    """

    # 先逐条样本把 district 变成 One-Hot 向量。
    # 这里直接拿当前训练集真实样本举例：
    #   训练集第1条样本的 district = "A"
    #   训练集里拟合出的 categories = ["A", "B", "C"]
    #   所以 One-Hot 结果是：
    #   [1, 0, 0]
    #
    #   训练集第3条样本的 district = "B"
    #   所以 One-Hot 结果是：
    #   [0, 1, 0]
    one_hot_rows = []
    for row in rows:
        one_hot_vector = transform_one_hot(row["district"], categories)
        one_hot_rows.append(one_hot_vector)

    # 把很多条 One-Hot 向量堆成一个二维矩阵。
    # 这里 rows 是训练集时，真实的 district 顺序是：
    #   A, A, B, B, C, A, C, B
    # 所以 one_hot_part 会变成：
    #   [
    #     [1, 0, 0],
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 1, 0],
    #     [0, 0, 1],
    #     [1, 0, 0],
    #     [0, 0, 1],
    #     [0, 1, 0],
    #   ]
    # 形状就是 (8, 3)
    one_hot_part = np.array(one_hot_rows, dtype=float)

    # np.hstack 的意思是 horizontal stack，也就是“按列方向拼接”。
    # 直接拿训练集第1条样本举真实例子：
    #
    #   第1条样本的标准化数值特征是：
    #   [-1.744, -1.134, 0.101, 0.666, -0.430, -0.775]
    #
    #   第1条样本的类别是 district="A"
    #   对应的 One-Hot 是：
    #   [1, 0, 0]
    #
    #   按列拼接后，第1条样本的最终特征就是：
    #   [-1.744, -1.134, 0.101, 0.666, -0.430, -0.775, 1, 0, 0]
    #
    # 整个矩阵层面上：
    #   scaled_numeric 的形状是 (8, 6)
    #   one_hot_part   的形状是 (8, 3)
    #   final_matrix   的形状就是 (8, 9)
    #
    # 能拼接的前提是两边行数相同，
    # 因为它们描述的是同一批样本，只是特征来源不同。
    final_matrix = np.hstack([scaled_numeric, one_hot_part])
    return final_matrix


X_train_final = build_final_feature_matrix(train_samples, X_train_scaled, district_categories)
X_valid_final = build_final_feature_matrix(valid_samples, X_valid_scaled, district_categories)
X_test_final = build_final_feature_matrix(test_samples, X_test_scaled, district_categories)

train_one_hot_part = np.array(
    [transform_one_hot(row["district"], district_categories) for row in train_samples],
    dtype=float,
)

print(f"训练集最终特征矩阵形状: {X_train_final.shape}")
print(f"验证集最终特征矩阵形状: {X_valid_final.shape}")
print(f"测试集最终特征矩阵形状: {X_test_final.shape}")
print("\n训练集的 One-Hot 类别特征矩阵:")
print(train_one_hot_part)
print("\n训练集标准化后的数值特征前2条样本:")
print(np.round(X_train_scaled[:2], 3))
print("\n训练集 One-Hot 类别特征前2条样本:")
print(train_one_hot_part[:2])
print("\n训练集最终特征矩阵前2条样本:")
print(np.round(X_train_final[:2], 3))
print("\n训练集第1条最终特征的拆解:")
print(f"  标准化数值部分 = {np.round(X_train_scaled[0], 3)}")
print(f"  One-Hot 类别部分 = {train_one_hot_part[0]}")
print(f"  最终拼接结果 = {np.round(X_train_final[0], 3)}")


# ============================================================
# 8. 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
这一章的数据预处理主线：

1. 先切分训练集 / 验证集 / 测试集
2. 在训练集上做特征工程与预处理参数拟合（fit）
3. 用训练集得到的规则去处理验证集和测试集（transform）
4. 最后得到适合模型学习的数值特征矩阵

本章核心结论：
- 原始字段通常不能直接喂给模型
- 类别特征要数值化（如 One-Hot）
- 数值特征经常要缩放（如标准化）
- 预处理也会发生数据泄漏
- “只在训练集上 fit” 是非常重要的基本原则
""")
