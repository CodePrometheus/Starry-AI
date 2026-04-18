"""
机器学习数学基础 (3) - 概率与统计
====================================
概率是 ML 的"语言"——模型的预测本质上是概率估计。
从基础概念出发，理解为什么 ML 离不开概率。
"""

import numpy as np


# ============================================================
# 1. 概率的直觉 — "不确定性的度量"
# ============================================================
# 概率 = 某件事发生的可能性，范围 [0, 1]
#   0 = 不可能发生
#   1 = 一定发生
#   0.5 = 一半一半
#
# ML 中的概率：
#   "这张图片是猫的概率是 0.92" → 模型很有把握
#   "这封邮件是垃圾邮件的概率是 0.51" → 模型不太确定

print("=" * 60)
print("1. 概率基础")
print("=" * 60)

# 用掷骰子来理解
# 掷一个公平骰子，每个面的概率 = 1/6
np.random.seed(42)
n_rolls = 10000
rolls = np.random.randint(1, 7, size=n_rolls)  # 1~6 的随机整数

print("掷骰子 10000 次的结果：")
for face in range(1, 7):
    count = np.sum(rolls == face)
    freq = count / n_rolls
    print(f"  面 {face}: 出现 {count} 次, 频率 = {freq:.4f} (理论概率 = {1/6:.4f})")
# 频率接近理论概率 → 这就是大数定律：试验次数越多，频率越接近真实概率


# ============================================================
# 2. 条件概率与贝叶斯定理
# ============================================================
# 条件概率 P(A|B) = "已知 B 发生了，A 发生的概率"
#
# 例子：垃圾邮件检测
#   P(垃圾|含"免费") = "已知邮件含'免费'，它是垃圾邮件的概率"
#
# 贝叶斯定理：
#   P(A|B) = P(B|A) × P(A) / P(B)
#
# 直觉：用新证据（B）来更新我们对 A 的信念
#   P(A) = 先验概率（看到证据之前的信念）
#   P(A|B) = 后验概率（看到证据之后的信念）
#
# 这是所有概率型 ML 模型的基础（朴素贝叶斯、贝叶斯网络等）

print("\n" + "=" * 60)
print("2. 条件概率与贝叶斯定理 — 垃圾邮件检测")
print("=" * 60)

# 场景：邮件分类
# 已知数据：
#   P(垃圾邮件) = 0.3              ← 30% 的邮件是垃圾邮件（先验）
#   P(含"免费"|垃圾邮件) = 0.8     ← 垃圾邮件中 80% 含"免费"
#   P(含"免费"|正常邮件) = 0.1     ← 正常邮件中 10% 含"免费"
#
# 问题：收到一封含"免费"的邮件，它是垃圾邮件的概率？

P_spam = 0.3                    # P(垃圾)
P_ham = 1 - P_spam              # P(正常) = 0.7
P_free_given_spam = 0.8         # P(含"免费"|垃圾)
P_free_given_ham = 0.1          # P(含"免费"|正常)

# 第1步：算 P(含"免费")（全概率公式）
# P(免费) = P(免费|垃圾)×P(垃圾) + P(免费|正常)×P(正常)
# 含义：不管是不是垃圾邮件，总共有多少邮件含"免费"？
P_free = P_free_given_spam * P_spam + P_free_given_ham * P_ham
print(f"P(含'免费') = {P_free_given_spam}×{P_spam} + {P_free_given_ham}×{P_ham} = {P_free}")

# 第2步：贝叶斯定理
# P(垃圾|免费) = P(免费|垃圾) × P(垃圾) / P(免费)
P_spam_given_free = P_free_given_spam * P_spam / P_free
print(f"\n贝叶斯定理：")
print(f"P(垃圾|含'免费') = P(含'免费'|垃圾) × P(垃圾) / P(含'免费')")
print(f"                 = {P_free_given_spam} × {P_spam} / {P_free}")
print(f"                 = {P_spam_given_free:.4f}")
print(f"\n→ 收到含'免费'的邮件，有 {P_spam_given_free:.1%} 概率是垃圾邮件")
print(f"→ 先验概率从 30% 更新到了 {P_spam_given_free:.1%}，这就是贝叶斯更新！")

# 用模拟验证
n_emails = 100000
is_spam = np.random.random(n_emails) < P_spam
has_free = np.where(
    is_spam,
    np.random.random(n_emails) < P_free_given_spam,  # 垃圾邮件含"免费"的概率
    np.random.random(n_emails) < P_free_given_ham     # 正常邮件含"免费"的概率
)
# 在含"免费"的邮件中，垃圾邮件的比例
simulated = np.sum(is_spam & has_free) / np.sum(has_free)
print(f"\n模拟验证（{n_emails} 封邮件）：")
print(f"  含'免费'的邮件中垃圾邮件比例 = {simulated:.4f}（理论值 {P_spam_given_free:.4f}）✓")


# ============================================================
# 3. 常见概率分布
# ============================================================
# 概率分布 = 描述随机变量所有可能取值及其概率的"全景图"
#
# ML 中为什么要关心分布？
#   - 数据服从什么分布，决定了用什么模型
#   - 损失函数的设计基于概率假设
#   - 正则化等价于给参数加上先验分布

print("\n" + "=" * 60)
print("3. 常见概率分布")
print("=" * 60)

# ----- 3.1 均匀分布 -----
# 每个值出现的概率相同
# 用途：随机初始化、随机采样
uniform_samples = np.random.uniform(0, 1, size=10000)
print(f"\n【均匀分布 Uniform(0,1)】")
print(f"  均值 = {uniform_samples.mean():.4f}（理论值 0.5）")
print(f"  标准差 = {uniform_samples.std():.4f}（理论值 {1/np.sqrt(12):.4f}）")

# ----- 3.2 正态（高斯）分布 -----
# 最重要的分布！"钟形曲线"
# 由均值 μ 和标准差 σ 完全决定
# 68% 的数据落在 μ±σ 内，95% 落在 μ±2σ 内
#
# ML 中的应用：
#   - 权重初始化（Xavier/He 初始化用正态分布）
#   - 回归模型假设误差服从正态分布
#   - 正则化等价于参数的正态先验
mu, sigma = 170, 10  # 均值170cm，标准差10cm（比如身高）
normal_samples = np.random.normal(mu, sigma, size=10000)
print(f"\n【正态分布 N(μ={mu}, σ={sigma})】模拟身高数据")
print(f"  均值 = {normal_samples.mean():.2f}（理论值 {mu}）")
print(f"  标准差 = {normal_samples.std():.2f}（理论值 {sigma}）")

# 验证 68-95-99.7 规则
within_1sigma = np.sum(np.abs(normal_samples - mu) <= sigma) / len(normal_samples)
within_2sigma = np.sum(np.abs(normal_samples - mu) <= 2 * sigma) / len(normal_samples)
within_3sigma = np.sum(np.abs(normal_samples - mu) <= 3 * sigma) / len(normal_samples)
print(f"  μ±1σ 内的比例 = {within_1sigma:.4f}（理论值 0.6827）")
print(f"  μ±2σ 内的比例 = {within_2sigma:.4f}（理论值 0.9545）")
print(f"  μ±3σ 内的比例 = {within_3sigma:.4f}（理论值 0.9973）")

# ----- 3.3 伯努利分布 -----
# 只有两个结果：成功(1) 或 失败(0)
# P(X=1) = p, P(X=0) = 1-p
# ML 中的应用：二分类问题的标签就是伯努利分布
p = 0.7  # 成功概率
bernoulli_samples = (np.random.random(10000) < p).astype(int)
print(f"\n【伯努利分布 Bernoulli(p={p})】")
print(f"  成功比例 = {bernoulli_samples.mean():.4f}（理论值 {p}）")
print(f"  → 二分类中的标签：是猫(1)/不是猫(0)")


# ============================================================
# 4. 均值、方差、标准差 — 描述数据的"位置"和"散布"
# ============================================================
# 均值 = 数据的"中心"在哪
# 方差 = 数据偏离中心的平均程度（越大越分散）
# 标准差 = 方差的平方根（和原始数据同单位，更直观）
#
# 在 ML 中：
#   - 特征标准化（减均值除标准差）是最常见的预处理
#   - 方差大的特征信息量可能更大（PCA 就是按方差排序）
#   - batch normalization 就是在每层做标准化

print("\n" + "=" * 60)
print("4. 均值、方差、标准差")
print("=" * 60)

# 两组考试成绩，均值相同但分布不同
scores_A = np.array([70, 72, 68, 71, 69])  # 分数很集中
scores_B = np.array([50, 90, 60, 80, 70])  # 分数很分散

print(f"班级 A 的成绩: {scores_A}")
print(f"班级 B 的成绩: {scores_B}")

# 均值
mean_A = scores_A.mean()
mean_B = scores_B.mean()
print(f"\n均值: A = {mean_A}, B = {mean_B}（均值相同！）")

# 方差 = 每个数据点与均值的差的平方的平均值
# var = (1/n) × Σ(xi - μ)²
var_A = scores_A.var()
var_B = scores_B.var()
print(f"方差: A = {var_A:.2f}, B = {var_B:.2f}（B 的方差大得多 → 成绩更分散）")

# 标准差 = √方差
std_A = scores_A.std()
std_B = scores_B.std()
print(f"标准差: A = {std_A:.2f}, B = {std_B:.2f}")
print(f"\n→ 虽然平均分一样，但 A 班很稳定（σ={std_A:.1f}），B 班差异很大（σ={std_B:.1f}）")


# ============================================================
# 5. 最大似然估计 (MLE) — ML 中"学习"的概率解释
# ============================================================
# 核心问题：给定观测数据，哪组参数最可能生成这些数据？
#
# 直觉：
#   你抛了 10 次硬币，7 次正面 3 次反面
#   问：这个硬币正面朝上的概率 p 最可能是多少？
#
#   如果 p=0.5 → 得到"7正3反"的概率 = C(10,7) × 0.5^7 × 0.5^3 = 0.117
#   如果 p=0.7 → 得到"7正3反"的概率 = C(10,7) × 0.7^7 × 0.3^3 = 0.267
#   如果 p=0.9 → 得到"7正3反"的概率 = C(10,7) × 0.9^7 × 0.1^3 = 0.057
#
#   p=0.7 使这个结果"最可能"出现 → 所以 MLE 估计 p=0.7
#
# 这和 ML 训练的联系：
#   损失函数（如交叉熵）本质上就是对数似然的负值
#   最小化损失 = 最大化似然 = 找最可能解释数据的参数

print("\n" + "=" * 60)
print("5. 最大似然估计 (MLE)")
print("=" * 60)

# 给定抛硬币的结果，找出最可能的正面概率 p
def coin_mle(flips):
    """
    参数：
        flips: numpy 数组，1 表示正面，0 表示反面
               例如 [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]

    返回：
        p_mle: 正面朝上的最大似然估计值

    提示：
        MLE 的结果其实很直觉——就是正面出现的频率
        p_mle = 正面次数 / 总次数
        想想为什么频率就是最大似然估计？
        因为这个 p 值能让观测到的数据出现的概率最大
    """
    count_1 = np.sum(flips)
    return count_1 / len(flips)


# 测试
np.random.seed(42)
flips = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1])  # 10 次抛硬币
print(f"抛硬币结果: {flips}")
print(f"正面 {np.sum(flips)} 次, 反面 {np.sum(1-flips)} 次")

p_mle = coin_mle(flips)
if p_mle is not None:
    print(f"\nMLE 估计 p = {p_mle:.4f}")

    # 画出似然函数，看看 p=0.7 是不是最大值
    p_range = np.linspace(0.01, 0.99, 100)
    n_heads = np.sum(flips)
    n_tails = np.sum(1 - flips)
    # 似然函数 L(p) = p^(正面次数) × (1-p)^(反面次数)
    # 用对数似然避免数值下溢：log L(p) = n_heads × log(p) + n_tails × log(1-p)
    log_likelihood = n_heads * np.log(p_range) + n_tails * np.log(1 - p_range)

    max_idx = np.argmax(log_likelihood)
    print(f"似然函数在 p = {p_range[max_idx]:.4f} 处取最大值（和 MLE 结果一致）")
    print(f"\n→ MLE 的本质：找到让数据出现概率最大的参数")
    print(f"→ ML 中的训练 = 最小化损失 = 最大化似然 = 找最可能的参数")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
概率与统计在 ML 中的对应关系：
┌─────────────────┬──────────────────────────────────────┐
│ 概率/统计概念    │ ML 中的应用                           │
├─────────────────┼──────────────────────────────────────┤
│ 条件概率         │ 分类器输出 P(类别|特征)               │
│ 贝叶斯定理       │ 朴素贝叶斯分类、贝叶斯优化            │
│ 正态分布         │ 权重初始化、误差假设、正则化先验       │
│ 伯努利分布       │ 二分类标签、逻辑回归                  │
│ 均值/标准差      │ 特征标准化、Batch Normalization       │
│ 最大似然估计     │ 损失函数的概率解释（交叉熵=负对数似然）│
└─────────────────┴──────────────────────────────────────┘
""")
