"""
机器学习基础 (16) - Self-Attention 与 Transformer Block
=========================================================
本脚本重点演示：

1. token embedding 和位置编码如何组合
2. 单头 self-attention 的 Q / K / V、分数矩阵与注意力权重
3. causal mask 为什么能把 attention 变成“只能看左边”
4. 一个最小 transformer block 的前向传播
"""

from __future__ import annotations

import numpy as np


np.set_printoptions(precision=3, suppress=True)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """支持矩阵的 softmax。"""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """最小可运行版 LayerNorm。"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def single_head_self_attention(
    x: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """最小单头 attention 前向传播。"""
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    scores = Q @ K.T / np.sqrt(Q.shape[-1])

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    attention_weights = softmax(scores, axis=-1)
    context = attention_weights @ V
    return Q, K, V, scores, attention_weights, context


def transformer_block(
    x: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """一个最小 transformer block：attention + 残差 + LN + FFN + 残差 + LN。"""
    Q, K, V, scores, attention_weights, context = single_head_self_attention(
        x=x,
        W_q=W_q,
        W_k=W_k,
        W_v=W_v,
        mask=mask,
    )
    attention_residual = x + context
    attention_norm = layer_norm(attention_residual)

    ff_hidden = np.maximum(0.0, attention_norm @ W1 + b1)
    ff_output = ff_hidden @ W2 + b2
    block_output = layer_norm(attention_norm + ff_output)

    return {
        "Q": Q,
        "K": K,
        "V": V,
        "scores": scores,
        "attention_weights": attention_weights,
        "context": context,
        "attention_norm": attention_norm,
        "ff_hidden": ff_hidden,
        "ff_output": ff_output,
        "block_output": block_output,
    }


def print_attention_row(tokens: list[str], attention_weights: np.ndarray, row_idx: int, title: str) -> None:
    """把某个 token 的注意力分布打印得更直观。"""
    row_text = ", ".join(
        f"{token}:{attention_weights[row_idx, idx]:.3f}"
        for idx, token in enumerate(tokens)
    )
    print(f"{title} -> {tokens[row_idx]} 的注意力分布: {row_text}")


print("=" * 60)
print("1. 准备一个最小输入序列")
print("=" * 60)

# 用一句真实中文短句做 attention 例子：
# “周末 去 上海 旅游”
# 这 4 个 token 的语义角色很不一样：时间 / 动作 / 地点 / 事件。
tokens = ["周末", "去", "上海", "旅游"]

token_embeddings = np.array([
    [0.0, 0.0, 0.2, 1.0],  # 周末：时间信息最强
    [0.0, 0.1, 1.1, 0.0],  # 去：动作
    [0.0, 1.2, 0.2, 0.0],  # 上海：地点
    [0.5, 0.7, 0.8, 0.0],  # 旅游：既和动作相关，也和地点相关
], dtype=float)

position_embeddings = np.array([
    [0.00, 0.00, 0.00, 0.00],
    [0.05, 0.00, 0.00, 0.02],
    [0.10, 0.00, 0.00, 0.04],
    [0.15, 0.00, 0.00, 0.06],
], dtype=float)

x = token_embeddings + position_embeddings

for idx, token in enumerate(tokens):
    print(f"{token}:")
    print(f"  token embedding = {token_embeddings[idx]}")
    print(f"  position embed  = {position_embeddings[idx]}")
    print(f"  输入向量 x      = {x[idx]}")


print("\n" + "=" * 60)
print("2. 从零计算单头 self-attention")
print("=" * 60)

# 为了让例子可解释，这里手动给一组固定权重，而不是随机初始化。
W_q = np.array([
    [0.8, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.3, 0.0],
    [0.0, 0.4, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.9],
], dtype=float)

W_k = np.array([
    [0.7, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.2, 0.0],
    [0.0, 0.3, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.8],
], dtype=float)

W_v = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.2, 1.0, 0.0, 0.0],
    [0.0, 0.2, 1.0, 0.0],
    [0.0, 0.0, 0.2, 1.0],
], dtype=float)

Q, K, V, scores, attention_weights, context = single_head_self_attention(
    x=x,
    W_q=W_q,
    W_k=W_k,
    W_v=W_v,
)

print(f"Q =\n{Q}")
print(f"\nK =\n{K}")
print(f"\nV =\n{V}")
print(f"\nscore 矩阵 =\n{scores}")
print(f"\nattention 权重 =\n{attention_weights}")
print(f"\ncontext 向量 =\n{context}")

print_attention_row(tokens, attention_weights, row_idx=1, title="完整 self-attention")
print_attention_row(tokens, attention_weights, row_idx=3, title="完整 self-attention")
print("\n可以把它理解成：")
print("  - token 先各自生成 query / key / value")
print("  - query 和所有 key 做相似度，得到注意力权重")
print("  - 再把所有 value 按权重加权求和，得到新的上下文表示")


print("\n" + "=" * 60)
print("3. causal mask：为什么语言模型不能偷看未来")
print("=" * 60)

causal_mask = np.tril(np.ones((len(tokens), len(tokens)), dtype=bool))
_, _, _, masked_scores, masked_attention, masked_context = single_head_self_attention(
    x=x,
    W_q=W_q,
    W_k=W_k,
    W_v=W_v,
    mask=causal_mask,
)

print(f"causal mask =\n{causal_mask.astype(int)}")
print(f"\n加 mask 后的 score 矩阵 =\n{masked_scores}")
print(f"\n加 mask 后的 attention 权重 =\n{masked_attention}")
print_attention_row(tokens, masked_attention, row_idx=1, title="causal self-attention")
print_attention_row(tokens, masked_attention, row_idx=2, title="causal self-attention")
print("\n例如第 2 个 token“去”只能看见自己和左边的“周末”，不能看右边的“上海”和“旅游”。")


print("\n" + "=" * 60)
print("4. 一个最小 transformer block")
print("=" * 60)

W1 = np.array([
    [0.8, -0.2, 0.3, 0.0, 0.5, 0.1],
    [0.0, 0.9, 0.2, 0.4, -0.1, 0.3],
    [0.2, 0.4, 0.9, 0.1, 0.2, 0.5],
    [0.1, 0.0, 0.2, 0.8, 0.3, 0.0],
], dtype=float)
b1 = np.array([0.0, 0.1, 0.0, 0.0, 0.1, 0.0], dtype=float)

W2 = np.array([
    [0.7, 0.0, 0.1, 0.0],
    [0.0, 0.6, 0.0, 0.1],
    [0.1, 0.0, 0.7, 0.2],
    [0.0, 0.1, 0.2, 0.6],
    [0.3, 0.1, 0.0, 0.1],
    [0.0, 0.2, 0.3, 0.0],
], dtype=float)
b2 = np.array([0.0, 0.05, 0.0, 0.0], dtype=float)

block_result = transformer_block(
    x=x,
    W_q=W_q,
    W_k=W_k,
    W_v=W_v,
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2,
    mask=causal_mask,
)

print(f"attention 后 + residual 再 LayerNorm 的结果 =\n{block_result['attention_norm']}")
print(f"\n前馈层隐藏表示 ff_hidden =\n{block_result['ff_hidden']}")
print(f"\ntransformer block 输出 =\n{block_result['block_output']}")

print("\n输入与输出形状：")
print(f"  输入 x.shape = {x.shape}")
print(f"  输出 y.shape = {block_result['block_output'].shape}")
print("形状保持不变，所以 transformer block 可以层层堆叠。")


print("\n" + "=" * 60)
print("5. TODO(human) 练习")
print("=" * 60)

# TODO(human):
# 1. 把 tokens 改成 ["下周", "去", "杭州", "出差"]，并自己重写 token_embeddings。
# 2. 把 causal_mask 去掉，再看第 3 部分里“去”的注意力分布会不会开始偷看右边 token。
print("练习 1：把句子改成“下周 去 杭州 出差”，自己重写 token embedding。")
print("练习 2：把 transformer_block 里的 mask=None，再观察 attention 权重如何变化。")
