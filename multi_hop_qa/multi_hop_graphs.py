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
            lambda thoughts, partId=part_id: [
                t for t in thoughts if t.state.get("part") == partId
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


def multiAgentGoT(num_branches: int = 4, local_branch_k: int = 2) -> operations.GraphOfOperations:
    """
    多智能体 GoT Hybrid：
    串行主链 + 每跳局部分叉（Retriever/Reasoner/Critic）+ 局部筛选，再进入下一跳。

    结构：
    Planner
      -> [Hop i: Retriever(k) -> Score -> KeepBestN(1)
                  -> Reasoner(k) -> Score -> KeepBestN(1)
                  -> Critic(k)   -> Score -> KeepBestN(1)] x N
      -> Aggregate(3) -> Score(LLM Judge) -> KeepBestN(1) -> GroundTruth
    """
    g = operations.GraphOfOperations()
    planner = operations.Generate(1, 1)
    g.append_operation(planner)

    num_branches = max(1, int(num_branches))
    local_branch_k = max(1, int(local_branch_k))
    previous = planner

    for _ in range(num_branches):
        retriever = operations.Generate(1, local_branch_k)
        retriever.add_predecessor(previous)
        g.add_operation(retriever)
        retriever_score = operations.Score(1, False, None)
        retriever_score.add_predecessor(retriever)
        g.add_operation(retriever_score)
        retriever_best = operations.KeepBestN(1, True)
        retriever_best.add_predecessor(retriever_score)
        g.add_operation(retriever_best)

        reasoner = operations.Generate(1, local_branch_k)
        reasoner.add_predecessor(retriever_best)
        g.add_operation(reasoner)
        reasoner_score = operations.Score(1, False, None)
        reasoner_score.add_predecessor(reasoner)
        g.add_operation(reasoner_score)
        reasoner_best = operations.KeepBestN(1, True)
        reasoner_best.add_predecessor(reasoner_score)
        g.add_operation(reasoner_best)

        critic = operations.Generate(1, local_branch_k)
        critic.add_predecessor(reasoner_best)
        g.add_operation(critic)
        critic_score = operations.Score(1, False, None)
        critic_score.add_predecessor(critic)
        g.add_operation(critic_score)
        critic_best = operations.KeepBestN(1, True)
        critic_best.add_predecessor(critic_score)
        g.add_operation(critic_best)

        previous = critic_best

    aggregate = operations.Aggregate(3)
    aggregate.add_predecessor(previous)
    g.add_operation(aggregate)
    g.append_operation(operations.Score(1, False, None))
    g.append_operation(operations.KeepBestN(1, True))
    g.append_operation(operations.GroundTruth(score.testMultiHop))
    return g
