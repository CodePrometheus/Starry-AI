"""
机器学习进阶 (19) - 读论文、看 ablation、做项目 framing
======================================================
这一章不讲某个单独算法，而是讲研究和落地时最容易混乱的三件事：

1. 论文应该怎么读，才能抓到真正有用的信息？
2. ablation 表应该怎么分析，才能知道“到底是哪一项起作用”？
3. 一个 LLM 小项目在开做之前，应该怎样先把 framing 说清楚？

脚本里的例子尽量贴近真实工作流：
- 论文示例：LoRA
- 项目示例：课程资料问答助手
"""

from __future__ import annotations

from dataclasses import dataclass


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


@dataclass
class PaperCard:
    title: str
    problem: str
    baseline: str
    method: str
    key_claim: str
    evidence: list[str]
    limitations: list[str]
    useful_tags: list[str]


@dataclass
class AblationRow:
    name: str
    changed_component: str
    answer_accuracy: float
    latency_ms: int
    strict_ablation: bool


@dataclass
class ProjectFrame:
    name: str
    target_user: str
    painful_moment: str
    input_example: str
    good_output_example: str
    available_data: str
    offline_metric: str
    online_guardrail: str
    latency_budget_ms: int
    minimum_demo: str
    kill_criteria: str


def paper_fit_report(paper: PaperCard, project_needs: list[str]) -> tuple[list[str], list[str]]:
    matched = [need for need in project_needs if need in paper.useful_tags]
    missing = [need for need in project_needs if need not in paper.useful_tags]
    return matched, missing


def print_ablation_analysis(rows: list[AblationRow], latency_budget_ms: int) -> None:
    baseline = rows[0]
    print(
        f"基线 = {baseline.name} | 准确率 = {baseline.answer_accuracy:.1%} | "
        f"延迟 = {baseline.latency_ms} ms"
    )

    best_single_change: AblationRow | None = None
    best_single_accuracy = baseline.answer_accuracy

    print("\n逐行看 delta：")
    for row in rows[1:]:
        delta_accuracy = row.answer_accuracy - baseline.answer_accuracy
        delta_latency = row.latency_ms - baseline.latency_ms
        within_budget = row.latency_ms <= latency_budget_ms

        print(
            f"- {row.name}: 改动 = {row.changed_component}, "
            f"准确率 delta = {delta_accuracy:+.1%}, "
            f"延迟 delta = {delta_latency:+d} ms, "
            f"预算内 = {within_budget}"
        )

        if row.strict_ablation and within_budget and row.answer_accuracy > best_single_accuracy:
            best_single_change = row
            best_single_accuracy = row.answer_accuracy

    if best_single_change is not None:
        print(
            "\n最值得先做的单变量实验："
            f"{best_single_change.name}，"
            f"因为它在预算内带来了最高的准确率。"
        )


def project_checklist(frame: ProjectFrame) -> list[tuple[str, bool]]:
    checks = [
        ("用户是谁", bool(frame.target_user)),
        ("痛点是否具体", bool(frame.painful_moment)),
        ("输入输出是否可举例", bool(frame.input_example and frame.good_output_example)),
        ("是否有可用数据", bool(frame.available_data)),
        ("是否定义离线指标", bool(frame.offline_metric)),
        ("是否定义线上护栏", bool(frame.online_guardrail)),
        ("是否有延迟预算", frame.latency_budget_ms > 0),
        ("是否知道最小 demo", bool(frame.minimum_demo)),
        ("是否定义 kill criteria", bool(frame.kill_criteria)),
    ]
    return checks


print_section("1. 论文阅读：先把 Paper Card 填出来")

lora_paper = PaperCard(
    title="LoRA: Low-Rank Adaptation of Large Language Models",
    problem="全参数微调成本高，显存占用大，难以快速给大模型做领域适配。",
    baseline="全参数微调，以及更早期的 adapter 类参数高效方法。",
    method="冻结原始权重，只训练低秩矩阵，让权重更新写成一个低秩增量。",
    key_claim="在多项任务上，LoRA 可以用很少的可训练参数接近全参数微调效果，并保持部署友好。",
    evidence=[
        "论文把 LoRA 和全参数微调做了任务效果对比。",
        "论文关注的不只是效果，还看了可训练参数规模和训练成本。",
        "方法可以在推理前合并回原权重，因此不会天然引入额外推理链路。",
    ],
    limitations=[
        "rank r 和插入位置要调，不是所有任务都能一把梭。",
        "如果任务主要瓶颈在检索、标注质量或评测方式，单靠 LoRA 可能帮不到核心问题。",
        "论文里的结论要结合你自己的模型规模、数据量和延迟预算再判断。",
    ],
    useful_tags=["低显存", "快速迭代", "领域适配", "部署友好"],
)

print(f"论文标题: {lora_paper.title}")
print(f"问题: {lora_paper.problem}")
print(f"对比基线: {lora_paper.baseline}")
print(f"方法: {lora_paper.method}")
print(f"核心 claim: {lora_paper.key_claim}")
print("\n证据应该盯什么：")
for evidence in lora_paper.evidence:
    print(f"- {evidence}")

print("\n局限不要跳过：")
for limitation in lora_paper.limitations:
    print(f"- {limitation}")

project_needs = ["低显存", "领域适配", "引用答案来源"]
matched_needs, missing_needs = paper_fit_report(lora_paper, project_needs)
print("\n如果你的项目需求是：", project_needs)
print(f"- 这篇论文直接匹配的需求: {matched_needs}")
print(f"- 这篇论文没有直接解决的需求: {missing_needs}")
print(
    "读论文不是只看“这个方法强不强”，"
    "而是要问：它解决的是不是我现在最贵的那个瓶颈。"
)


print_section("2. 看 ablation：分清单变量实验和系统组合实验")

ablation_rows = [
    AblationRow(
        name="Base prompt",
        changed_component="无",
        answer_accuracy=0.54,
        latency_ms=180,
        strict_ablation=True,
    ),
    AblationRow(
        name="+ JSON 输出约束",
        changed_component="只改输出格式提示",
        answer_accuracy=0.58,
        latency_ms=182,
        strict_ablation=True,
    ),
    AblationRow(
        name="+ 4 条 few-shot",
        changed_component="只增加示例",
        answer_accuracy=0.64,
        latency_ms=195,
        strict_ablation=True,
    ),
    AblationRow(
        name="+ 检索",
        changed_component="只增加 RAG 检索",
        answer_accuracy=0.77,
        latency_ms=290,
        strict_ablation=True,
    ),
    AblationRow(
        name="+ LoRA",
        changed_component="只增加参数高效微调",
        answer_accuracy=0.72,
        latency_ms=205,
        strict_ablation=True,
    ),
    AblationRow(
        name="+ 检索 + LoRA",
        changed_component="系统组合，不是严格单变量",
        answer_accuracy=0.80,
        latency_ms=305,
        strict_ablation=False,
    ),
]

latency_budget_ms = 320
print("教学化实验表：课程资料问答助手，指标是答案准确率@1")
for row in ablation_rows:
    print(
        f"- {row.name:<18} | 准确率 = {row.answer_accuracy:.1%} | "
        f"延迟 = {row.latency_ms} ms | strict_ablation = {row.strict_ablation}"
    )

print()
print_ablation_analysis(ablation_rows, latency_budget_ms=latency_budget_ms)
print(
    "\n这里最重要的读法是："
    "先看单变量实验，确认到底是哪一个部件真有贡献；"
    "最后再看系统组合，判断上线版本值不值得。"
)


print_section("3. 项目 framing：先把问题定义清楚，再写代码")

qa_project = ProjectFrame(
    name="课程资料问答助手",
    target_user="正在复习大模型课程的工程师",
    painful_moment="资料分散在讲义和笔记里，复习时总要翻很多页找答案。",
    input_example="LoRA 和全参数微调最大的区别是什么？",
    good_output_example="LoRA 冻结原权重，只训练低秩增量矩阵；因此显存和训练成本更低，并附上引用段落。",
    available_data="120 份课程讲义、20 份 FAQ、500 条历史问答。",
    offline_metric="答案准确率@1 + 引用命中率",
    online_guardrail="没有证据时必须回答“我不知道”，并返回空引用。",
    latency_budget_ms=350,
    minimum_demo="先做一个 CLI demo：检索 top-3 段落，再生成带引用的答案。",
    kill_criteria="如果两周后准确率仍低于 70%，或引用命中率低于 95%，就暂停继续堆功能。",
)

print(f"项目名: {qa_project.name}")
print(f"目标用户: {qa_project.target_user}")
print(f"具体痛点: {qa_project.painful_moment}")
print(f"输入示例: {qa_project.input_example}")
print(f"理想输出: {qa_project.good_output_example}")
print(f"可用数据: {qa_project.available_data}")
print(f"离线指标: {qa_project.offline_metric}")
print(f"线上护栏: {qa_project.online_guardrail}")
print(f"延迟预算: {qa_project.latency_budget_ms} ms")
print(f"最小 demo: {qa_project.minimum_demo}")
print(f"终止条件: {qa_project.kill_criteria}")

print("\n项目 checklist：")
all_checks = project_checklist(qa_project)
for item, passed in all_checks:
    print(f"- {item}: {'OK' if passed else 'MISSING'}")

print(
    "\n这一步的目标不是把计划写得很宏大，"
    "而是把“谁会用、什么算好、什么情况该停”先说清楚。"
)


print_section("4. 一个最小研究/落地分析流")

analysis_flow = [
    "先写出 1 个具体用户场景和 3 条真实问题，不要先讨论模型名。",
    "做最强可复现 baseline，例如 prompt-only 或 prompt + retrieval。",
    "定义 1 个主指标和 1~2 个护栏指标，不要一开始就追很多分数。",
    "ablation 一次只改 1 个变量，例如只加检索、只加 few-shot、只加 LoRA。",
    "如果单变量实验都没有稳定收益，不要急着堆组合系统。",
    "只有当效果、延迟、护栏都过线，才值得进入更完整的产品化。",
]

for index, step in enumerate(analysis_flow, start=1):
    print(f"{index}. {step}")

print(
    "\n结合上面的教学数据，一个合理顺序通常是："
    "先做 retrieval baseline，再评估 LoRA 是否值得进入第二阶段。"
)


print_section("5. TODO(human) 练习")

print("TODO(human):")
print("1. 把你自己的一个想法，按 ProjectFrame 这 10 个字段写出来。")
print("2. 给 ablation_rows 再加一行 '+ rerank'，判断它是不是严格单变量实验。")
print("3. 再找一篇论文，补一张新的 Paper Card，并写出它没有解决的瓶颈。")
