"""
控制器模块包：管理操作图的执行流程。

Controller 类负责：
- 按拓扑顺序执行操作图中的操作
- 协调语言模型、提示词生成器和解析器
- 输出执行结果
"""

from .controller import Controller
