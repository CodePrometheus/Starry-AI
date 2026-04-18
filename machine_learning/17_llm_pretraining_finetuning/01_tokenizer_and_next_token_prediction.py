"""
机器学习进阶 (17) - Tokenizer、Next-Token Prediction 与玩具语言模型
===================================================================
这一章用纯标准库把 LLM 最核心的三个直觉跑通：

1. 文本为什么不能直接喂给模型，而要先切成 token？
2. 预训练里的 "预测下一个 token" 到底在学什么？
3. 一个极小的 toy language model 是如何工作的？

额外补一层微调直觉：
- 同一个基础模型，继续喂一小批领域数据后，输出分布会发生偏移。
"""

from __future__ import annotations

from collections import Counter, defaultdict


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def count_adjacent_pairs(token_sequences: list[list[str]]) -> Counter[tuple[str, str]]:
    """统计所有相邻 token 对，模拟 BPE/merge tokenizer 的第一步。"""
    pair_counts: Counter[tuple[str, str]] = Counter()
    for sequence in token_sequences:
        for left, right in zip(sequence, sequence[1:]):
            pair_counts[(left, right)] += 1
    return pair_counts


def merge_pair(token_sequences: list[list[str]], pair: tuple[str, str]) -> list[list[str]]:
    """把给定 pair 合并成一个新 token。"""
    merged_token = "".join(pair)
    merged_sequences: list[list[str]] = []

    for sequence in token_sequences:
        new_sequence: list[str] = []
        index = 0
        while index < len(sequence):
            if (
                index < len(sequence) - 1
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                new_sequence.append(merged_token)
                index += 2
            else:
                new_sequence.append(sequence[index])
                index += 1
        merged_sequences.append(new_sequence)

    return merged_sequences


def tokenize_with_vocab(text: str, vocab_tokens: list[str]) -> list[str]:
    """
    用最长匹配做一个极简 tokenizer。

    真实 tokenizer 更复杂，但这个版本足够演示：
    - 高频短语会变成更长的 token
    - 未登录字符会退化成 <UNK>
    """
    regular_tokens = [
        token for token in vocab_tokens if token not in {"<BOS>", "<EOS>", "<UNK>"}
    ]
    regular_tokens.sort(key=len, reverse=True)

    output: list[str] = []
    index = 0
    while index < len(text):
        matched = None
        for token in regular_tokens:
            if text.startswith(token, index):
                matched = token
                break

        if matched is None:
            output.append("<UNK>")
            index += 1
        else:
            output.append(matched)
            index += len(matched)

    return output


def build_bigram_counts(token_sequences: list[list[str]]) -> dict[str, Counter[str]]:
    """构造一个最简单的 bigram 语言模型：只看当前 token 预测下一个 token。"""
    transitions: dict[str, Counter[str]] = defaultdict(Counter)

    for sequence in token_sequences:
        tokens = ["<BOS>"] + sequence + ["<EOS>"]
        for current_token, next_token in zip(tokens, tokens[1:]):
            transitions[current_token][next_token] += 1

    return dict(transitions)


def format_counter(counter: Counter[str]) -> str:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    total = sum(counter.values())
    formatted = [
        f"{token}: {count}/{total} = {count / total:.2f}" for token, count in items
    ]
    return " | ".join(formatted)


def greedy_generate(
    prompt_tokens: list[str],
    transitions: dict[str, Counter[str]],
    max_new_tokens: int = 6,
) -> list[str]:
    """
    用贪心策略生成文本。

    注意：这是 bigram 模型，所以它只看最后一个 token。
    这正好能暴露 toy LM 的局限。
    """
    generated = prompt_tokens[:]
    current_token = prompt_tokens[-1] if prompt_tokens else "<BOS>"

    for _ in range(max_new_tokens):
        next_counter = transitions.get(current_token)
        if not next_counter:
            break

        next_token = sorted(
            next_counter.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

        if next_token == "<EOS>":
            break

        generated.append(next_token)
        current_token = next_token

    return generated


print_section("1. tokenizer 直觉：模型看到的是 token，不是整句字符串")

raw_corpus = [
    "大模型喜欢数据",
    "大模型喜欢推理",
    "小模型也喜欢数据",
]

char_level_sequences = [list(text) for text in raw_corpus]
for text, sequence in zip(raw_corpus, char_level_sequences):
    print(f"原始文本: {text}")
    print(f"字符级切分: {sequence}")

pair_counts = count_adjacent_pairs(char_level_sequences)
top_pairs = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))[:6]
print("\n字符语料里最常见的相邻 pair：")
for pair, count in top_pairs:
    print(f"  {pair} -> {count} 次")

# 这里手动选择几个高频 pair 来模拟 merge。
# 真实 BPE 会循环地自动挑选高频 pair，我们这里只做教学版。
chosen_merges = [("模", "型"), ("喜", "欢"), ("数", "据")]
merged_sequences = char_level_sequences
for pair in chosen_merges:
    merged_sequences = merge_pair(merged_sequences, pair)

print("\n做了 3 次 merge 之后：")
for before, after in zip(char_level_sequences, merged_sequences):
    print(f"  {before} -> {after} | 长度 {len(before)} -> {len(after)}")

print(
    "\n直觉：高频片段会被合并成更长的 token，"
    "这样序列更短，模型也更容易复用已经见过的模式。"
)


print_section("2. token id：模型最终处理的是整数序列")

special_tokens = ["<BOS>", "<EOS>", "<UNK>"]
vocab = sorted({token for sequence in merged_sequences for token in sequence})
vocab_tokens = special_tokens + vocab
token_to_id = {token: index for index, token in enumerate(vocab_tokens)}
id_to_token = {index: token for token, index in token_to_id.items()}

example_text = "大模型喜欢数据"
example_tokens = tokenize_with_vocab(example_text, vocab_tokens)
example_ids = [token_to_id[token] for token in example_tokens]

print(f"词表大小 = {len(vocab_tokens)}")
print(f"词表内容 = {vocab_tokens}")
print(f"\n句子: {example_text}")
print(f"token 序列: {example_tokens}")
print(f"id 序列: {example_ids}")
print(
    f"模型不会直接理解 '{example_text}' 这串字符，"
    f"它只会看到像 {example_ids} 这样的整数序列。"
)


print_section("3. next-token prediction：预训练的核心监督信号")

transitions = build_bigram_counts(merged_sequences)

for sequence in merged_sequences:
    tokens = ["<BOS>"] + sequence + ["<EOS>"]
    pairs = list(zip(tokens, tokens[1:]))
    print(f"序列 {tokens}")
    print(f"训练对 {pairs}")

focus_token = "喜欢"
print(f"\n当前 token = '{focus_token}' 时，下一个 token 的经验分布：")
print(format_counter(transitions[focus_token]))
print(
    "\n这就是 next-token prediction 的最小版本："
    "看到当前上下文后，给下一个 token 一个概率分布。"
)


print_section("4. toy language model：从 prompt 一步步往后生成")

prompt_a = tokenize_with_vocab("大模型", vocab_tokens)
prompt_b = tokenize_with_vocab("小模型", vocab_tokens)

generated_a = greedy_generate(prompt_a, transitions)
generated_b = greedy_generate(prompt_b, transitions)

print(f"prompt = {prompt_a} -> 生成结果 = {generated_a}")
print(f"prompt = {prompt_b} -> 生成结果 = {generated_b}")
print(
    "\n注意：'大模型' 和 '小模型' 的最后一个 token 都是 '模型'，"
    "而 bigram 模型只看最后一个 token，"
    "所以它会把两个 prompt 处理得几乎一样。"
)
print(
    "这就是 toy LM 的局限：上下文窗口太短时，"
    "模型会丢失更长距离的信息。"
)


print_section("5. 微调直觉：继续喂一小批领域数据，分布就会偏移")

domain_sequences = [
    ["大", "模型", "喜欢", "安全"],
    ["模型", "喜欢", "安全"],
    ["中文", "模型", "喜欢", "安全"],
]

base_likes = transitions["喜欢"]
finetuned_sequences = merged_sequences + domain_sequences
finetuned_transitions = build_bigram_counts(finetuned_sequences)
finetuned_likes = finetuned_transitions["喜欢"]

print("微调前，'喜欢' 的下一个 token 分布：")
print(format_counter(base_likes))
print("\n加入 3 条安全领域小语料后：")
print(format_counter(finetuned_likes))

finetuned_prompt = tokenize_with_vocab("大模型", vocab_tokens)
finetuned_generation = greedy_generate(finetuned_prompt, finetuned_transitions)
print(f"\n微调后 prompt = {finetuned_prompt} -> 生成结果 = {finetuned_generation}")
print(
    "直觉：参数更新以后，模型更倾向于输出新数据里更常见的模式。"
    "真实 LLM 微调当然比这里复杂得多，但方向感就是这样来的。"
)


print_section("6. TODO(human) 练习")

print("TODO(human):")
print("1. 把 ('推', '理') 也加入 merge 列表，再观察 token 长度是否继续缩短。")
print("2. 把 bigram 改成 trigram，让 '小 模型' 后面更容易接 '也'。")
print("3. 自己增加 3 条新语料，看看微调后 '喜欢' 的分布如何变化。")
