"""
机器学习进阶 (18) - LLM 评估与推理基础
======================================
这一章聚焦三个最常见的问题：

1. 怎么用 perplexity 理解语言模型在验证集上的好坏？
2. greedy、temperature、top-k 采样到底会怎样改变输出？
3. 推理效率为什么总在谈 KV cache、batching、量化？
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def build_vocab(*corpora: list[list[str]]) -> list[str]:
    vocab = {"<BOS>", "<EOS>", "<UNK>"}
    for corpus in corpora:
        for sentence in corpus:
            vocab.update(sentence)
    return sorted(vocab)


def train_bigram_model(
    corpus: list[list[str]],
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sentence in corpus:
        tokens = ["<BOS>"] + sentence + ["<EOS>"]
        for current_token, next_token in zip(tokens, tokens[1:]):
            counts[current_token][next_token] += 1
    return dict(counts)


def conditional_probability(
    model: dict[str, Counter[str]],
    current_token: str,
    next_token: str,
    vocab_size: int,
    alpha: float = 1.0,
) -> float:
    """
    用 add-one smoothing 避免未见过的转移直接变成 0 概率。

    真实 LLM 不会这样算，但这里足够帮助我们理解 perplexity。
    """
    next_counts = model.get(current_token, Counter())
    total = sum(next_counts.values())
    return (next_counts[next_token] + alpha) / (total + alpha * vocab_size)


def evaluate_perplexity(
    name: str,
    model: dict[str, Counter[str]],
    validation_corpus: list[list[str]],
    vocab: list[str],
) -> float:
    vocab_size = len(vocab)
    total_negative_log_prob = 0.0
    total_predictions = 0

    print(f"模型: {name}")
    for sentence_index, sentence in enumerate(validation_corpus, start=1):
        tokens = ["<BOS>"] + sentence + ["<EOS>"]
        sentence_nll = 0.0

        print(f"  验证句子 {sentence_index}: {tokens}")
        for current_token, next_token in zip(tokens, tokens[1:]):
            probability = conditional_probability(
                model=model,
                current_token=current_token,
                next_token=next_token,
                vocab_size=vocab_size,
            )
            negative_log_prob = -math.log(probability)
            sentence_nll += negative_log_prob
            total_negative_log_prob += negative_log_prob
            total_predictions += 1

            print(
                "    "
                f"P({next_token}|{current_token}) = {probability:.4f}, "
                f"-log p = {negative_log_prob:.4f}"
            )

        sentence_avg_nll = sentence_nll / (len(tokens) - 1)
        sentence_ppl = math.exp(sentence_avg_nll)
        print(
            f"    句子平均 NLL = {sentence_avg_nll:.4f}, "
            f"句子 perplexity = {sentence_ppl:.4f}"
        )

    average_nll = total_negative_log_prob / total_predictions
    perplexity = math.exp(average_nll)
    print(f"  总平均 NLL = {average_nll:.4f}")
    print(f"  总 perplexity = {perplexity:.4f}")
    return perplexity


def softmax(logits: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    adjusted_logits = {
        token: logit / temperature
        for token, logit in logits.items()
    }
    max_logit = max(adjusted_logits.values())
    exp_values = {
        token: math.exp(logit - max_logit)
        for token, logit in adjusted_logits.items()
    }
    total = sum(exp_values.values())
    return {
        token: value / total
        for token, value in exp_values.items()
    }


def apply_top_k(probabilities: dict[str, float], top_k: int | None) -> dict[str, float]:
    if top_k is None or top_k >= len(probabilities):
        return probabilities

    kept_items = sorted(
        probabilities.items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]
    total = sum(probability for _, probability in kept_items)
    return {token: probability / total for token, probability in kept_items}


def pretty_distribution(probabilities: dict[str, float]) -> str:
    items = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    return " | ".join(f"{token}: {probability:.3f}" for token, probability in items)


def sample_from_distribution(
    probabilities: dict[str, float],
    rng: random.Random,
) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for token, probability in sorted(
        probabilities.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        cumulative += probability
        if threshold <= cumulative:
            return token
    return list(probabilities)[-1]


def generate_with_strategy(
    logit_table: dict[str, dict[str, float]],
    strategy: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 7,
    max_new_tokens: int = 6,
) -> list[str]:
    rng = random.Random(seed)
    current_token = "<BOS>"
    output: list[str] = []

    for _ in range(max_new_tokens):
        logits = logit_table[current_token]
        probabilities = softmax(logits, temperature=temperature)
        probabilities = apply_top_k(probabilities, top_k=top_k)

        if strategy == "greedy":
            next_token = sorted(
                probabilities.items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0]
        else:
            next_token = sample_from_distribution(probabilities, rng)

        if next_token == "<EOS>":
            break

        output.append(next_token)
        current_token = next_token

    return output


def attention_scores_without_cache(prompt_len: int, new_tokens: int) -> int:
    return sum((prompt_len + step) ** 2 for step in range(new_tokens))


def attention_scores_with_cache(prompt_len: int, new_tokens: int) -> int:
    prefill = prompt_len ** 2
    decode = sum(prompt_len + step for step in range(new_tokens))
    return prefill + decode


def padding_waste(sequence_lengths: list[int]) -> int:
    max_len = max(sequence_lengths)
    return max_len * len(sequence_lengths) - sum(sequence_lengths)


def model_weight_memory_gib(params_in_billions: float, bytes_per_param: float) -> float:
    total_bytes = params_in_billions * 1_000_000_000 * bytes_per_param
    return total_bytes / (1024 ** 3)


print_section("1. perplexity 直觉：同样的验证句子，哪个模型更不惊讶？")

good_train_corpus = [
    ["今天", "学习", "大模型"],
    ["今天", "学习", "推理"],
    ["模型", "推理", "需要", "显存"],
    ["批量", "推理", "提升", "吞吐"],
    ["推理", "速度", "影响", "体验"],
    ["贪心", "解码", "结果", "稳定"],
]

bad_train_corpus = [
    ["今天", "去", "公园"],
    ["晚饭", "想", "吃", "面"],
    ["周末", "一起", "看", "电影"],
    ["天气", "不错", "适合", "散步"],
    ["跑步", "之后", "想", "休息"],
]

validation_corpus = [
    ["今天", "学习", "推理"],
    ["推理", "速度", "影响", "体验"],
]

vocab = build_vocab(good_train_corpus, bad_train_corpus, validation_corpus)
good_model = train_bigram_model(good_train_corpus)
bad_model = train_bigram_model(bad_train_corpus)

good_ppl = evaluate_perplexity("领域相关语料训练的模型", good_model, validation_corpus, vocab)
print()
bad_ppl = evaluate_perplexity("领域无关语料训练的模型", bad_model, validation_corpus, vocab)

print("\n结论：")
print(f"- 相关语料模型 perplexity = {good_ppl:.4f}")
print(f"- 无关语料模型 perplexity = {bad_ppl:.4f}")
print(
    "perplexity 越低，表示模型对验证集越不惊讶。"
    "但它只衡量概率拟合，不直接等于“回答一定更有用”。"
)


print_section("2. greedy、temperature、top-k：采样策略如何改变输出")

toy_logit_table = {
    "<BOS>": {"今天": 3.0, "现在": 1.8, "周末": 0.9},
    "今天": {"学习": 3.2, "训练": 2.6, "摸鱼": 1.0, "休息": 0.3},
    "现在": {"开始": 2.4, "讨论": 2.0, "休息": 1.0},
    "周末": {"休息": 2.2, "学习": 1.2, "<EOS>": 0.5},
    "学习": {"大模型": 2.9, "概率": 1.5, "<EOS>": 0.4},
    "训练": {"模型": 2.6, "代码": 1.8, "<EOS>": 0.5},
    "开始": {"推理": 2.3, "<EOS>": 0.4},
    "讨论": {"采样": 2.1, "<EOS>": 0.5},
    "休息": {"<EOS>": 2.0},
    "摸鱼": {"<EOS>": 1.8},
    "大模型": {"很": 2.5, "需要": 2.0, "<EOS>": 0.3},
    "概率": {"<EOS>": 2.1},
    "模型": {"<EOS>": 2.0, "优化": 1.0},
    "代码": {"<EOS>": 1.9},
    "推理": {"<EOS>": 2.2},
    "采样": {"<EOS>": 2.1},
    "很": {"有趣": 2.3, "贵": 1.0, "<EOS>": 0.4},
    "需要": {"数据": 2.1, "显存": 1.8, "<EOS>": 0.3},
    "有趣": {"<EOS>": 2.3},
    "贵": {"<EOS>": 1.7},
    "数据": {"<EOS>": 2.2},
    "显存": {"<EOS>": 2.2},
    "优化": {"<EOS>": 1.9},
}

today_logits = toy_logit_table["今天"]
today_default = softmax(today_logits, temperature=1.0)
today_cold = softmax(today_logits, temperature=0.7)
today_hot = softmax(today_logits, temperature=1.5)
today_top_k = apply_top_k(today_default, top_k=2)

print("上下文 token = '今天' 时的候选分布：")
print(f"- temperature = 1.0: {pretty_distribution(today_default)}")
print(f"- temperature = 0.7: {pretty_distribution(today_cold)}")
print(f"- temperature = 1.5: {pretty_distribution(today_hot)}")
print(f"- top-k = 2      : {pretty_distribution(today_top_k)}")

greedy_output = generate_with_strategy(toy_logit_table, strategy="greedy")
temp_output = generate_with_strategy(
    toy_logit_table,
    strategy="sample",
    temperature=1.5,
    seed=7,
)
top_k_output = generate_with_strategy(
    toy_logit_table,
    strategy="sample",
    temperature=1.0,
    top_k=2,
    seed=7,
)

print("\n固定随机种子 seed=7 后，不同策略生成结果：")
print(f"- greedy                -> {greedy_output}")
print(f"- temperature = 1.5     -> {temp_output}")
print(f"- temperature = 1.0 top-k = 2 -> {top_k_output}")
print(
    "\n直觉：greedy 最稳定，但容易保守；"
    "temperature 越高，分布越平，越容易采到低概率 token；"
    "top-k 会先砍掉尾部候选，再在头部里采样。"
)


print_section("3. 推理效率直觉：KV cache、batching、量化")

prompt_len = 8
new_tokens = 6
no_cache_scores = attention_scores_without_cache(prompt_len, new_tokens)
with_cache_scores = attention_scores_with_cache(prompt_len, new_tokens)

print(f"prompt 长度 = {prompt_len}，继续生成 {new_tokens} 个 token")
print(f"- 不用 KV cache，粗略 attention 计算量 ~ {no_cache_scores}")
print(f"- 使用 KV cache，粗略 attention 计算量 ~ {with_cache_scores}")
print(
    f"- 近似减少到原来的 {with_cache_scores / no_cache_scores:.2%}"
)

uniform_batch = [18, 18, 17, 17]
mixed_batch = [18, 40, 75, 120]
print("\nbatch padding 浪费对比：")
print(
    f"- 长度接近的 batch {uniform_batch} -> padding 浪费 {padding_waste(uniform_batch)} token"
)
print(
    f"- 长度差很大的 batch {mixed_batch} -> padding 浪费 {padding_waste(mixed_batch)} token"
)

params_b = 7.0
fp16_memory = model_weight_memory_gib(params_b, bytes_per_param=2.0)
int8_memory = model_weight_memory_gib(params_b, bytes_per_param=1.0)
int4_memory = model_weight_memory_gib(params_b, bytes_per_param=0.5)
print("\n7B 模型不同精度下的权重显存占用（只看参数，不含 KV cache/激活值）：")
print(f"- FP16: {fp16_memory:.2f} GiB")
print(f"- INT8: {int8_memory:.2f} GiB")
print(f"- INT4: {int4_memory:.2f} GiB")
print(
    "这就是为什么部署时经常会讨论“能不能量化、能不能开 KV cache、"
    "batch 要怎么分桶”。"
)


print_section("4. TODO(human) 练习")

print("TODO(human):")
print("1. 自己实现 top-p (nucleus) sampling，并和 top-k 对比。")
print("2. 把 prompt_len 改成 128、512、2048，观察 KV cache 的收益如何变化。")
print("3. 把 validation_corpus 换成你自己的领域句子，再比较 perplexity。")
