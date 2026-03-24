from graph_of_thoughts import operations

import score


def io() -> operations.GraphOfOperations:
    """IO：一次生成。流程为 Generate -> Score -> GroundTruth。"""
    g = operations.GraphOfOperations()
    g.append_operation(operations.Generate(1, 1))
    g.append_operation(operations.Score(1, False, score.scoreMultiHop))
    g.append_operation(operations.GroundTruth(score.testMultiHop))
    return g


def cot() -> operations.GraphOfOperations:
    """CoT：与 IO 的操作图相同，但提示词要求给出逐步推理后再输出答案。"""
    g = operations.GraphOfOperations()
    g.append_operation(operations.Generate(1, 1))
    g.append_operation(operations.Score(1, False, score.scoreMultiHop))
    g.append_operation(operations.GroundTruth(score.testMultiHop))
    return g


def tot() -> operations.GraphOfOperations:
    """ToT：多候选生成 -> 打分 -> 保留最优 -> 再生成/精化一次 -> 再打分 -> 保留最优。"""
    g = operations.GraphOfOperations()
    g.append_operation(operations.Generate(1, 5))
    g.append_operation(operations.Score(1, False, score.scoreMultiHop))
    k1 = operations.KeepBestN(1, True)
    g.append_operation(k1)
    g.append_operation(operations.Generate(1, 3))
    g.append_operation(operations.Score(1, False, score.scoreMultiHop))
    g.append_operation(operations.KeepBestN(1, True))
    g.append_operation(operations.GroundTruth(score.testMultiHop))
    return g


def got() -> operations.GraphOfOperations:
    """GoT：重叠区间分组摘要 -> 两路局部结论 -> 各路 Score -> Aggregate(3 条候选) -> 再评分 -> 保留最优 -> GroundTruth。"""
    g = operations.GraphOfOperations()
    plans = operations.Generate(1, 1)
    g.append_operation(plans)

    for i in range(2):
        part_id = f"Group {i}"
        sel = operations.Selector(
            lambda thoughts, part_id=part_id: [
                t for t in thoughts if t.state.get("part") == part_id
            ]
        )
        sel.add_predecessor(plans)
        g.add_operation(sel)
        gen = operations.Generate(1, 1)
        gen.add_predecessor(sel)
        g.add_operation(gen)
        branch_score = operations.Score(1, False, score.scoreMultiHop)
        branch_score.add_predecessor(gen)
        g.add_operation(branch_score)

    g.append_operation(operations.Aggregate(3))
    g.append_operation(operations.Score(1, False, score.scoreMultiHop))
    g.append_operation(operations.KeepBestN(1, True))
    g.append_operation(operations.GroundTruth(score.testMultiHop))
    return g


def multiAgentGoT() -> operations.GraphOfOperations:
    """
    多智能体 GoT：
    Planner -> (Retriever -> Reasoner -> Critic) x up to 4 sub-questions
    -> Aggregate -> Score(LLM Judge) -> KeepBestN -> GroundTruth
    """
    g = operations.GraphOfOperations()
    planner = operations.Generate(1, 1)
    g.append_operation(planner)

    critic_leaves = []
    for sub_id in range(4):
        sel = operations.Selector(
            lambda thoughts, sid=sub_id: [t for t in thoughts if t.state.get("sub_id") == sid]
        )
        sel.add_predecessor(planner)
        g.add_operation(sel)

        retriever = operations.Generate(1, 1)
        retriever.add_predecessor(sel)
        g.add_operation(retriever)

        reasoner = operations.Generate(1, 1)
        reasoner.add_predecessor(retriever)
        g.add_operation(reasoner)

        critic = operations.Generate(1, 1)
        critic.add_predecessor(reasoner)
        g.add_operation(critic)

        critic_score = operations.Score(1, False, None)
        critic_score.add_predecessor(critic)
        g.add_operation(critic_score)
        critic_leaves.append(critic_score)

    aggregate = operations.Aggregate(3)
    for leaf in critic_leaves:
        aggregate.add_predecessor(leaf)
    g.add_operation(aggregate)
    g.append_operation(operations.Score(1, False, None))
    g.append_operation(operations.KeepBestN(1, True))
    g.append_operation(operations.GroundTruth(score.testMultiHop))
    return g
