"""
提示词生成器模块包：定义构造 LLM 提示词的接口。

Prompter 基类定义了以下提示词生成方法：
- generate_prompt: 生成新思维的提示词
- aggregation_prompt: 聚合思维的提示词
- score_prompt: 评分的提示词
- validation_prompt: 验证的提示词
- improve_prompt: 改进的提示词
"""

from .prompter import Prompter
