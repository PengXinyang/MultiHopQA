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
    多智能体 GoT Hybrid (真正的图结构 + 全局动态回溯)：
    Planner 将问题拆解为 N 个子问题，每个子问题启动一个并行的推理分支。
    在每个分支中：Retriever 提取证据 -> Reasoner 生成局部结论 -> Critic 检验。
    如果 Critic 判定不相关 (REJECT)，则触发回溯，打回给该分支的 Retriever 重新生成。
    最后，Aggregate 节点等待所有分支成功完成，汇总所有证据生成最终答案。
    """
    g = operations.GraphOfOperations()
    planner = operations.Generate(1, 1)
    g.append_operation(planner)

    num_branches = max(1, int(num_branches))
    local_branch_k = max(1, int(local_branch_k))
    
    branch_leaves = []

    for i in range(num_branches):
        # Selector: 挑选出分配给当前分支的子问题 (sub_id == i)
        sel = operations.Selector(
            lambda thoughts, idx=i: [t for t in thoughts if t.state.get("sub_id") == idx]
        )
        sel.add_predecessor(planner)
        g.add_operation(sel)

        retriever = operations.Generate(1, local_branch_k)
        retriever.add_predecessor(sel)
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

        # Critic 检验与回溯
        critic_verify = operations.CriticVerifyAndBacktrack(target_backtrack_op=retriever, max_retries=2)
        critic_verify.add_predecessor(reasoner_best)
        g.add_operation(critic_verify)

        branch_leaves.append(critic_verify)

    # 聚合所有并行分支的有效结论
    aggregate = operations.Aggregate(1)
    for leaf in branch_leaves:
        aggregate.add_predecessor(leaf)
    g.add_operation(aggregate)

    final_score = operations.Score(1, False, None)
    final_score.add_predecessor(aggregate)
    g.add_operation(final_score)

    keep_best = operations.KeepBestN(1, True)
    keep_best.add_predecessor(final_score)
    g.add_operation(keep_best)

    gt = operations.GroundTruth(score.testMultiHop)
    gt.add_predecessor(keep_best)
    g.add_operation(gt)

    return g
