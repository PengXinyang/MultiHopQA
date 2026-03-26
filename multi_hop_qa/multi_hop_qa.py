# 多跳推理示例：在 GoT 框架下对多文档问答进行 IO / CoT / ToT / GoT 多种推理方式。

import argparse
import logging
import os
import random
import time
from typing import List, Callable

import utils
from graph_of_thoughts import language_models, operations
from multi_hop_graphs import io, cot, tot, got, multiAgentGoT
from multi_hop_parser import MultiHopParser
from multi_hop_prompter import MultiHopPrompter
from role_aware_lm import RoleAwareLM


# --- Run ---


def run(
        data_ids: List[int],
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        role_model_names: dict[str, str],
        dataset: str,
        data_path: str = None,
        max_samples: int = 100,
) -> float:
    """
    加载多跳问答数据集（HotpotQA / MuSiQue），对指定样本和指定方法运行 GoT 框架，
    并将每次运行的 GRS（Graph Reasoning State）输出到 `results/` 目录下。
    
    Args:
        data_ids: 要运行的样本索引列表，None 或空列表表示全部
        methods: 方法列表，每个方法返回一个 GraphOfOperations
        budget: 预算（美元），超出后停止
        role_model_names: 语言模型名称
        dataset: 运行哪个数据集
        data_path: 数据集路径，None 则使用默认 HotpotQA
        max_samples: 最大加载样本数
    
    Returns:
        spent: 实际花费（美元）
    """
    # 1. 解析数据路径
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "hotpotQA", "hotpot_dev_distractor_v1.json"
        )
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))

    # 2. 加载数据
    data = utils.loadMultiHopData(data_path, max_samples=max_samples)
    if not data:
        raise FileNotFoundError(f"No data loaded from {data_path}")

    # 3. 选择样本
    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected = [data[i] for i in data_ids if i < len(data)]

    # 4. 创建运行目录和配置
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    run_dir = utils.setupRunDirectory(dataset, results_base_dir=results_dir, lm_name=role_model_names["default"],
                                      methods=methods, config_extra={
            "data_path": data_path,
            "data_ids": data_ids[:len(selected)],
            "budget": budget,
            "max_samples": max_samples,
        })

    # 5. 获取语言模型配置路径
    config_lm_path = utils.getLmConfigPath(os.path.dirname(__file__))

    # 6. 运行实验
    spent = 0.0
    prompter = MultiHopPrompter()
    parser = MultiHopParser()

    for idx, item in enumerate(selected, start=1):
        print(f"正在运行第 {idx}/{len(selected)} 个问题，_id={item.get('_id', '')}")
        if budget <= 0:
            logging.error("Budget depleted, stopping.")
            break

        for method in methods:
            print(f"  方法: {method.__name__}")
            if budget <= 0:
                break

            # 创建语言模型实例
            if method.__name__.startswith("multiAgentGoT"):
                role_to_lm = {}
                for role, model_name in role_model_names.items():
                    if model_name == "__lite__":
                        role_to_lm[role] = language_models.LightweightModelGroup(
                            config_lm_path, cache=True, retries_per_model=3
                        )
                    elif model_name == "__heavy__":
                        role_to_lm[role] = language_models.HeavyModelGroup(
                            config_lm_path, cache=True, retries_per_model=3
                        )
                    else:
                        role_to_lm[role] = language_models.build_language_model(
                            config_lm_path,
                            model_name=model_name,
                            cache=True,
                        )
                lm = RoleAwareLM(role_to_lm=role_to_lm, default_role="default")
            else:
                model_name = role_model_names["default"]
                if model_name == "__lite__":
                    lm = language_models.LightweightModelGroup(
                        config_lm_path, cache=True, retries_per_model=3
                    )
                elif model_name == "__heavy__":
                    lm = language_models.HeavyModelGroup(
                        config_lm_path, cache=True, retries_per_model=3
                    )
                else:
                    lm = language_models.build_language_model(
                        config_lm_path,
                        model_name=model_name,
                        cache=True,
                    )

            # 执行单个方法
            cost = utils.runSingleMethod(item=item, method=method, lm=lm, prompter=prompter, parser=parser,
                                         run_dir=run_dir)

            budget -= cost
            spent += cost

    return spent


if __name__ == "__main__":
    # 默认入口：随机跑样本，方法为 io/cot/tot/got/multiAgentGoT，预算为 5 美元
    # 支持的数据集：hotpotqa, musique_ans, musique_full
    parser = argparse.ArgumentParser(description="多跳问答 GoT 实验")
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        choices=["hotpotqa", "musique_ans", "musique_full"],
                        help="数据集名称")
    parser.add_argument("--budget", type=float, default=5.0,
                        help="预算（美元）")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="随机抽取的样本数")
    parser.add_argument(
        "--sample_id",
        type=str,
        default="",
        help="按样本 id/_id 指定运行单题（优先于随机抽样）",
    )
    args = parser.parse_args()

    # 数据集路径和大小映射
    DATASET_CONFIG = {
        "hotpotqa": {
            "path": os.path.join(os.path.dirname(__file__), "..", "dataset", "hotpotQA",
                                 "hotpot_dev_distractor_v1.json"),
            "size": 7405,
        },
        "musique_ans": {
            "path": os.path.join(os.path.dirname(__file__), "..", "dataset", "MuSiQue", "musique_ans_v1.0_dev.jsonl"),
            "size": 2417,
        },
        "musique_full": {
            "path": os.path.join(os.path.dirname(__file__), "..", "dataset", "MuSiQue", "musique_full_v1.0_dev.jsonl"),
            "size": 4834,
        },
    }

    config = DATASET_CONFIG[args.dataset]
    data_path = config["path"]
    len_data = config["size"]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件不存在: {data_path}")

    seed = int(time.time())
    # seed = 42
    random.seed(seed)
    # samples = random.sample(range(len_data), min(args.num_samples, len_data))
    # 支持按 id/_id 定位样本（用于复现实验与调试）
    if args.sample_id and args.sample_id.strip():
        target = args.sample_id.strip()
        data = utils.loadMultiHopData(data_path, max_samples=len_data)
        hits = []
        for i, item in enumerate(data):
            sid = str(item.get("_id") or item.get("id") or "")
            if sid == target:
                hits.append(i)
        if not hits:
            raise ValueError(f"未在数据集中找到 sample_id={target}")
        samples = hits
    else:
        samples = random.sample(range(len_data), min(args.num_samples, len_data))
    #approaches = [io, cot, tot, got, multiAgentGoT]
    approaches = [multiAgentGoT]

    # 角色模型分配（方案A：一个角色一个智能体/模型实例）
    # 角色模型分配
    # - "__lite__": 轻量模型轮换池（见 graph_of_thoughts.language_models.rotating）
    # - "__heavy__": 复杂模型轮换池（见 graph_of_thoughts.language_models.rotating）
    # - 其它字符串：单一模型（config.json 的 key）
    role_model_names = {
        "planner": "__heavy__",
        "retriever": "__lite__",
        "reasoner": "__heavy__",
        "critic": "__heavy__",
        "default": "__heavy__",
    }

    print(f"数据集: {args.dataset}")
    print(f"语言模型: {role_model_names}")
    print(f"样本数: {len(samples)}")
    print(f"预算: ${args.budget}")

    spent = run(
        samples,
        approaches,
        args.budget,
        role_model_names,
        args.dataset,
        data_path=data_path,
        max_samples=len_data,
    )
    logging.info("Spent %s out of %s budget.", spent, args.budget)
