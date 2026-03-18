"""
统一的多跳问答数据集加载器

支持的数据集：
- HotpotQA (distractor / fullwiki)
- MuSiQue (ans / full)

使用方式：
    from data_loader import load_dataset, DatasetType

    # 加载 HotpotQA
    data = load_dataset("path/to/hotpot.json", DatasetType.HOTPOTQA)

    # 加载 MuSiQue
    data = load_dataset("path/to/musique.jsonl", DatasetType.MUSIQUE)

    # 自动检测
    data = load_dataset("path/to/data", DatasetType.AUTO)
"""

import json
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


class DatasetType(Enum):
    """数据集类型枚举"""
    AUTO = auto()       # 自动检测
    HOTPOTQA = auto()   # HotpotQA
    MUSIQUE = auto()    # MuSiQue


@dataclass
class Paragraph:
    """统一的段落格式"""
    title: str                          # 段落标题
    text: str                           # 段落文本（完整段落或拼接后的句子）
    sentences: List[str] = field(default_factory=list)  # 句子列表（HotpotQA 格式）
    is_supporting: bool = False         # 是否是支持段落
    idx: int = 0                        # 段落在 context 中的索引


@dataclass
class DecompositionStep:
    """问题分解步骤（MuSiQue 特有）"""
    step_id: str                        # 步骤 ID
    question: str                       # 子问题
    answer: str                         # 子答案
    paragraph_support_idx: Optional[int] = None  # 支持段落索引


@dataclass
class SupportingFact:
    """支持事实"""
    title: str                          # 文档标题
    sent_idx: int                       # 句子索引（HotpotQA）或段落索引（MuSiQue）


@dataclass
class MultiHopSample:
    """统一的多跳问答样本格式"""
    id: str                             # 样本 ID
    question: str                       # 问题
    answer: str                         # 标准答案
    paragraphs: List[Paragraph]         # 段落列表
    supporting_facts: List[SupportingFact]  # 支持事实列表
    
    # 可选字段
    answer_aliases: List[str] = field(default_factory=list)  # 答案别名（MuSiQue）
    answerable: bool = True             # 是否可回答（MuSiQue full）
    question_decomposition: List[DecompositionStep] = field(default_factory=list)  # 问题分解（MuSiQue）
    
    # 元数据
    dataset_type: str = ""              # 数据集类型
    question_type: str = ""             # 问题类型（HotpotQA: comparison/bridge）
    level: str = ""                     # 难度级别（HotpotQA: easy/medium/hard）
    num_hops: int = 2                   # 推理跳数

    def toDict(self) -> Dict[str, Any]:
        """转换为字典格式（兼容现有代码）"""
        return {
            "_id": self.id,
            "question": self.question,
            "answer": self.answer,
            "context": [
                [p.title, p.sentences if p.sentences else [p.text]]
                for p in self.paragraphs
            ],
            "supporting_facts": [
                [sf.title, sf.sent_idx] for sf in self.supporting_facts
            ],
            "answer_aliases": self.answer_aliases,
            "answerable": self.answerable,
            "question_decomposition": [
                {
                    "id": step.step_id,
                    "question": step.question,
                    "answer": step.answer,
                    "paragraph_support_idx": step.paragraph_support_idx,
                }
                for step in self.question_decomposition
            ],
            "type": self.question_type,
            "level": self.level,
            "num_hops": self.num_hops,
            "dataset_type": self.dataset_type,
        }

    def getContextText(self, with_indices: bool = False) -> str:
        """获取格式化的上下文文本"""
        parts = []
        for i, p in enumerate(self.paragraphs):
            if with_indices:
                if p.sentences:
                    for idx, sent in enumerate(p.sentences):
                        parts.append(f"[{i}][{p.title}][{idx}] {sent}")
                else:
                    parts.append(f"[{i}][{p.title}] {p.text}")
            else:
                text = " ".join(p.sentences) if p.sentences else p.text
                parts.append(f"[{p.title}]\n{text}")
        return "\n\n".join(parts) if not with_indices else "\n".join(parts)


# ============== HotpotQA 加载器 ==============

def loadHotpotQA(path: str, max_samples: Optional[int] = None) -> List[MultiHopSample]:
    """加载 HotpotQA 数据集"""
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    if max_samples is not None:
        raw_data = raw_data[:max_samples]
    
    samples = []
    for item in raw_data:
        # 解析 context
        paragraphs = []
        title_to_idx = {}
        for idx, ctx in enumerate(item.get("context", [])):
            title = ctx[0]
            sentences = ctx[1] if len(ctx) > 1 else []
            title_to_idx[title] = idx
            paragraphs.append(Paragraph(
                title=title,
                text=" ".join(sentences) if isinstance(sentences, list) else str(sentences),
                sentences=sentences if isinstance(sentences, list) else [sentences],
                is_supporting=False,
                idx=idx,
            ))
        
        # 解析 supporting_facts 并标记支持段落
        supporting_facts = []
        for sf in item.get("supporting_facts", []):
            title, sent_idx = sf[0], sf[1]
            supporting_facts.append(SupportingFact(title=title, sent_idx=sent_idx))
            # 标记对应段落为支持段落
            if title in title_to_idx:
                paragraphs[title_to_idx[title]].is_supporting = True
        
        sample = MultiHopSample(
            id=item.get("_id", ""),
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            paragraphs=paragraphs,
            supporting_facts=supporting_facts,
            answerable=True,
            dataset_type="hotpotqa",
            question_type=item.get("type", ""),
            level=item.get("level", ""),
            num_hops=2,
        )
        samples.append(sample)
    
    return samples


# ============== MuSiQue 加载器 ==============

def loadMusique(path: str, max_samples: Optional[int] = None) -> List[MultiHopSample]:
    """加载 MuSiQue 数据集（JSONL 格式）"""
    samples = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if max_samples is not None and line_num >= max_samples:
                break
            
            item = json.loads(line.strip())
            
            # 解析 paragraphs
            paragraphs = []
            for idx, p in enumerate(item.get("paragraphs", [])):
                paragraphs.append(Paragraph(
                    title=p.get("title", ""),
                    text=p.get("paragraph_text", ""),
                    sentences=[],  # MuSiQue 不提供句子级分割
                    is_supporting=p.get("is_supporting", False),
                    idx=idx,
                ))
            
            # 解析 question_decomposition
            decomposition = []
            supporting_facts = []
            for step in item.get("question_decomposition", []):
                decomposition.append(DecompositionStep(
                    step_id=str(step.get("id", "")),
                    question=step.get("question", ""),
                    answer=step.get("answer", ""),
                    paragraph_support_idx=step.get("paragraph_support_idx"),
                ))
                # 从 decomposition 提取 supporting_facts
                para_idx = step.get("paragraph_support_idx")
                if para_idx is not None and para_idx < len(paragraphs):
                    supporting_facts.append(SupportingFact(
                        title=paragraphs[para_idx].title,
                        sent_idx=para_idx,  # MuSiQue 用段落索引
                    ))
            
            # 从 id 推断跳数（2hop__, 3hop__, 4hop__）
            sample_id = item.get("id", "")
            num_hops = 2
            if sample_id.startswith("2hop"):
                num_hops = 2
            elif sample_id.startswith("3hop"):
                num_hops = 3
            elif sample_id.startswith("4hop"):
                num_hops = 4
            
            sample = MultiHopSample(
                id=sample_id,
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                paragraphs=paragraphs,
                supporting_facts=supporting_facts,
                answer_aliases=item.get("answer_aliases", []),
                answerable=item.get("answerable", True),
                question_decomposition=decomposition,
                dataset_type="musique",
                num_hops=num_hops,
            )
            samples.append(sample)
    
    return samples


# ============== 自动检测数据集类型 ==============

def detectDatasetType(path: str) -> DatasetType:
    """根据文件扩展名和内容自动检测数据集类型"""
    if path.endswith(".jsonl"):
        return DatasetType.MUSIQUE
    
    if path.endswith(".json"):
        # 尝试读取第一条数据判断
        with open(path, "r", encoding="utf-8") as f:
            content = f.read(1000)  # 读取前 1000 个字符
            if '"_id"' in content and '"context"' in content:
                return DatasetType.HOTPOTQA
            if '"paragraphs"' in content and '"question_decomposition"' in content:
                return DatasetType.MUSIQUE
    
    # 默认按 HotpotQA 处理
    return DatasetType.HOTPOTQA


# ============== 统一加载接口 ==============

def loadDataset(
    path: str,
    dataset_type: DatasetType = DatasetType.AUTO,
    max_samples: Optional[int] = None,
    return_dict: bool = False,
) -> List:
    """
    统一的数据集加载接口
    
    Args:
        path: 数据集文件路径
        dataset_type: 数据集类型，默认自动检测
        max_samples: 最大加载样本数，None 表示全部加载
        return_dict: 是否返回字典格式（兼容现有代码），默认返回 MultiHopSample 对象
    
    Returns:
        样本列表（MultiHopSample 对象或字典）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集文件不存在: {path}")
    
    # 自动检测数据集类型
    if dataset_type == DatasetType.AUTO:
        dataset_type = detectDatasetType(path)
    
    # 加载数据
    if dataset_type == DatasetType.HOTPOTQA:
        samples = loadHotpotQA(path, max_samples)
    elif dataset_type == DatasetType.MUSIQUE:
        samples = loadMusique(path, max_samples)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    # 转换为字典格式
    if return_dict:
        return [s.toDict() for s in samples]
    
    return samples


def loadHotpotData(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    兼容旧版 utils.load_hotpot_data 的接口
    
    Args:
        path: 数据集文件路径
        max_samples: 最大加载样本数
    
    Returns:
        字典格式的样本列表
    """
    return loadDataset(path, DatasetType.AUTO, max_samples, return_dict=True)


# ============== 便捷函数 ==============

def getSupportingParagraphs(sample: MultiHopSample) -> List[Paragraph]:
    """获取所有支持段落"""
    return [p for p in sample.paragraphs if p.is_supporting]


def getDistractorParagraphs(sample: MultiHopSample) -> List[Paragraph]:
    """获取所有干扰段落"""
    return [p for p in sample.paragraphs if not p.is_supporting]


def filterByAnswerable(samples: List[MultiHopSample], answerable: bool = True) -> List[MultiHopSample]:
    """按可回答性过滤样本"""
    return [s for s in samples if s.answerable == answerable]


def filterByNumHops(samples: List[MultiHopSample], num_hops: int) -> List[MultiHopSample]:
    """按跳数过滤样本"""
    return [s for s in samples if s.num_hops == num_hops]


def filterByQuestionType(samples: List[MultiHopSample], question_type: str) -> List[MultiHopSample]:
    """按问题类型过滤样本（HotpotQA: comparison/bridge）"""
    return [s for s in samples if s.question_type == question_type]


# ============== 统计函数 ==============

def getDatasetStats(samples: List[MultiHopSample]) -> Dict[str, Any]:
    """获取数据集统计信息"""
    if not samples:
        return {}
    
    num_samples = len(samples)
    num_answerable = sum(1 for s in samples if s.answerable)
    avg_paragraphs = sum(len(s.paragraphs) for s in samples) / num_samples
    avg_supporting = sum(len(getSupportingParagraphs(s)) for s in samples) / num_samples
    
    # 跳数分布
    hop_dist = {}
    for s in samples:
        hop_dist[s.num_hops] = hop_dist.get(s.num_hops, 0) + 1
    
    # 问题类型分布（HotpotQA）
    type_dist = {}
    for s in samples:
        if s.question_type:
            type_dist[s.question_type] = type_dist.get(s.question_type, 0) + 1
    
    return {
        "total_samples": num_samples,
        "answerable_samples": num_answerable,
        "unanswerable_samples": num_samples - num_answerable,
        "avg_paragraphs": round(avg_paragraphs, 2),
        "avg_supporting_paragraphs": round(avg_supporting, 2),
        "hop_distribution": hop_dist,
        "question_type_distribution": type_dist,
        "dataset_type": samples[0].dataset_type if samples else "",
    }


# ============== 测试代码 ==============

if __name__ == "__main__":
    import sys
    
    # 测试 HotpotQA
    hotpot_path = "../hotpotQA/hotpot_dev_distractor_v1.json"
    if os.path.exists(hotpot_path):
        print("=" * 50)
        print("测试 HotpotQA 加载")
        print("=" * 50)
        samples = loadDataset(hotpot_path, max_samples=5)
        print(f"加载了 {len(samples)} 条样本")
        print(f"统计信息: {getDatasetStats(samples)}")
        print(f"\n第一条样本:")
        print(f"  ID: {samples[0].id}")
        print(f"  问题: {samples[0].question}")
        print(f"  答案: {samples[0].answer}")
        print(f"  段落数: {len(samples[0].paragraphs)}")
        print(f"  支持段落数: {len(getSupportingParagraphs(samples[0]))}")
    
    # 测试 MuSiQue
    musique_path = "../MuSiQue/musique_ans_v1.0_dev.jsonl"
    if os.path.exists(musique_path):
        print("\n" + "=" * 50)
        print("测试 MuSiQue 加载")
        print("=" * 50)
        samples = loadDataset(musique_path, max_samples=5)
        print(f"加载了 {len(samples)} 条样本")
        print(f"统计信息: {getDatasetStats(samples)}")
        print(f"\n第一条样本:")
        print(f"  ID: {samples[0].id}")
        print(f"  问题: {samples[0].question}")
        print(f"  答案: {samples[0].answer}")
        print(f"  段落数: {len(samples[0].paragraphs)}")
        print(f"  跳数: {samples[0].num_hops}")
        if samples[0].question_decomposition:
            print(f"  问题分解:")
            for step in samples[0].question_decomposition:
                print(f"    - {step.question} -> {step.answer}")
