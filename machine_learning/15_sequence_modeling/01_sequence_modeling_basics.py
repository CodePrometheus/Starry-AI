"""
机器学习基础 (15) - 序列建模入门
====================================
本脚本重点演示：

1. 什么是序列数据，为什么顺序很重要
2. token 序列如何表示成 id、one-hot、embedding
3. 如何把序列切成 next-token 训练样本
4. 一个最小 RNN 如何逐步更新隐藏状态
"""

from __future__ import annotations

import numpy as np


np.set_printoptions(precision=3, suppress=True)


def softmax(x: np.ndarray) -> np.ndarray:
    """对一维向量做 softmax。"""
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def decode_ids(token_ids: list[int], id_to_token: dict[int, str]) -> list[str]:
    """把 id 序列还原成 token 序列。"""
    return [id_to_token[token_id] for token_id in token_ids]


def build_next_token_examples(
    encoded_sequence: list[int],
    context_size: int,
    pad_id: int,
) -> list[tuple[list[int], int]]:
    """把一条序列切成“前文 -> 下一个 token”的监督学习样本。"""
    pairs: list[tuple[list[int], int]] = []
    for target_pos in range(1, len(encoded_sequence)):
        context = encoded_sequence[max(0, target_pos - context_size):target_pos]
        if len(context) < context_size:
            context = [pad_id] * (context_size - len(context)) + context
        target = encoded_sequence[target_pos]
        pairs.append((context, target))
    return pairs


print("=" * 60)
print("1. 序列数据长什么样")
print("=" * 60)

# 用真实一点的电商浏览会话做例子。
# 这些数据不是“无序特征表”，而是明确有先后关系的行为序列。
shopping_sessions = [
    ["打开APP", "搜索_跑鞋", "查看_跑鞋", "加入购物车", "下单"],
    ["打开APP", "搜索_咖啡豆", "查看_咖啡豆", "加入购物车", "下单"],
    ["打开APP", "搜索_瑜伽垫", "查看_瑜伽垫", "收藏", "退出"],
]

for idx, session in enumerate(shopping_sessions, start=1):
    print(f"会话 {idx}: {' -> '.join(session)}")

print("\n同样包含“查看_跑鞋”和“加入购物车”，顺序不同，含义也会不同。")
print("这就是序列建模和普通表格建模的核心区别。")


print("\n" + "=" * 60)
print("2. token -> id -> one-hot -> embedding")
print("=" * 60)

special_tokens = ["<PAD>", "<BOS>", "<EOS>"]
vocab_tokens = sorted({token for session in shopping_sessions for token in session})
vocab = special_tokens + vocab_tokens

token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

print("词表：")
for token, token_id in token_to_id.items():
    print(f"  {token:>10s} -> {token_id}")

encoded_sessions: list[list[int]] = []
for session in shopping_sessions:
    encoded = [token_to_id["<BOS>"]]
    encoded.extend(token_to_id[token] for token in session)
    encoded.append(token_to_id["<EOS>"])
    encoded_sessions.append(encoded)

print("\n第 1 条会话加上起止标记后的 token：")
print(decode_ids(encoded_sessions[0], id_to_token))
print("对应的 id 序列：")
print(encoded_sessions[0])

sample_token = "加入购物车"
sample_token_id = token_to_id[sample_token]
sample_one_hot = np.eye(len(vocab), dtype=float)[sample_token_id]

# embedding 是可学习的稠密表示。
# 这里只是演示 lookup，所以先随机初始化一个 embedding 表。
embedding_dim = 4
rng = np.random.default_rng(7)
embedding_table = rng.normal(loc=0.0, scale=0.6, size=(len(vocab), embedding_dim))

print(f"\n样例 token: {sample_token}")
print(f"id = {sample_token_id}")
print(f"one-hot = {sample_one_hot}")
print(f"embedding = {np.round(embedding_table[sample_token_id], 3)}")

first_session_embeddings = embedding_table[encoded_sessions[0]]
print(f"\n第 1 条会话的 embedding 序列形状 = {first_session_embeddings.shape}")
print("每一行就是一个 token 的稠密向量表示。")


print("\n" + "=" * 60)
print("3. 切出 next-token 训练样本")
print("=" * 60)

context_size = 3
all_pairs: list[tuple[list[int], int]] = []
for encoded in encoded_sessions:
    all_pairs.extend(
        build_next_token_examples(
            encoded_sequence=encoded,
            context_size=context_size,
            pad_id=token_to_id["<PAD>"],
        )
    )

print(f"上下文长度 context_size = {context_size}")
print(f"一共构造出 {len(all_pairs)} 条训练样本。\n")

print("前 6 条训练样本：")
for context_ids, target_id in all_pairs[:6]:
    context_tokens = decode_ids(context_ids, id_to_token)
    target_token = id_to_token[target_id]
    print(f"  context = {context_tokens} -> target = {target_token}")

print("\n拿第 1 条会话举例：")
print("  ['<PAD>', '<PAD>', '<BOS>'] -> '打开APP'")
print("意思是：还没有看到真正行为时，模型先学会从 <BOS> 开始。")
print("  ['<BOS>', '打开APP', '搜索_跑鞋'] -> '查看_跑鞋'")
print("意思是：前文已经出现“打开APP -> 搜索_跑鞋”，下一个 token 应该更像“查看_跑鞋”。")


print("\n" + "=" * 60)
print("4. 最小 RNN：隐藏状态如何随序列更新")
print("=" * 60)

# 这里只做前向传播，不做训练。
# 重点是看 h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h) 如何把“过去的信息”带到下一步。
hidden_size = 5
rnn_rng = np.random.default_rng(42)
W_xh = rnn_rng.normal(scale=0.45, size=(embedding_dim, hidden_size))
W_hh = rnn_rng.normal(scale=0.45, size=(hidden_size, hidden_size))
b_h = np.zeros(hidden_size)
W_hy = rnn_rng.normal(scale=0.45, size=(hidden_size, len(vocab)))
b_y = np.zeros(len(vocab))


def top_tokens(probabilities: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
    """显示 top-k token，忽略 <PAD>，避免输出太干扰。"""
    candidate_ids = [
        token_id
        for token, token_id in token_to_id.items()
        if token != "<PAD>"
    ]
    sorted_ids = sorted(candidate_ids, key=lambda idx: probabilities[idx], reverse=True)[:top_k]
    return [(id_to_token[idx], float(probabilities[idx])) for idx in sorted_ids]


def run_simple_rnn(sequence_tokens: list[str], title: str) -> np.ndarray:
    """逐 token 跑一个最小 RNN，观察隐藏状态变化。"""
    hidden = np.zeros(hidden_size)
    print(f"\n{title}")
    print(f"输入序列: {' -> '.join(sequence_tokens)}")

    for step, token in enumerate(sequence_tokens, start=1):
        token_id = token_to_id[token]
        x_t = embedding_table[token_id]
        hidden = np.tanh(x_t @ W_xh + hidden @ W_hh + b_h)
        logits = hidden @ W_hy + b_y
        probabilities = softmax(logits)
        top3 = top_tokens(probabilities)
        top3_text = ", ".join(f"{name}:{prob:.3f}" for name, prob in top3)

        print(f"  step {step}: 输入 token = {token}")
        print(f"           hidden = {np.round(hidden, 3)}")
        print(f"           当前状态读出的 top-3 下一步 token = {top3_text}")

    return hidden


normal_order = ["打开APP", "搜索_跑鞋", "查看_跑鞋", "加入购物车"]
changed_order = ["打开APP", "查看_跑鞋", "搜索_跑鞋", "加入购物车"]

normal_hidden = run_simple_rnn(normal_order, "序列 A：正常浏览顺序")
changed_hidden = run_simple_rnn(changed_order, "序列 B：打乱中间两步")

hidden_distance = np.linalg.norm(normal_hidden - changed_hidden)
print("\n最终隐藏状态对比：")
print(f"  序列 A 的最终 hidden = {np.round(normal_hidden, 3)}")
print(f"  序列 B 的最终 hidden = {np.round(changed_hidden, 3)}")
print(f"  两个最终状态的 L2 距离 = {hidden_distance:.4f}")
print("即使最后一个 token 都是“加入购物车”，前面的顺序变了，RNN 状态也会变。")


print("\n" + "=" * 60)
print("5. TODO(human) 练习")
print("=" * 60)

# TODO(human):
# 1. 把 changed_order 改成 ["打开APP", "搜索_跑鞋", "加入购物车", "查看_跑鞋"]，
#    再运行脚本，观察最终 hidden state 会怎么变。
# 2. 把 context_size 从 3 改成 2，重新看第 3 部分输出，感受“上下文窗口缩短”会丢掉哪些信息。
print("练习 1：把 changed_order 换成另一种顺序，重新观察最终 hidden state。")
print("练习 2：把 context_size 从 3 改成 2，再看 next-token 样本的上下文有什么变化。")

