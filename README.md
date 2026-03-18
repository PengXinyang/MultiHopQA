# Multi-hop QA (多跳推理)

本目录在 Graph of Thoughts (GoT) 框架下实现**多跳问答**示例，使用 Hotpot QA 风格数据（多文档 + 需跨文档推理的问题）。

## 任务与数据

- **输入**：一个问题 + 多篇文档（context，如 10 篇）。
- **输出**：答案（answer），可选支持事实（supporting_facts）。
- **数据格式**：与 Hotpot 一致，每条约含 `question`、`context`、`answer`、`supporting_facts`、`_id` 等。
- **数据路径**：默认使用项目上级目录下的 `hotpot/hotpot_dev_distractor_v1.json`；可在 `run()` 中通过 `data_path` 指定其它 JSON。

## 推理方式

- **IO**：单次生成，直接根据全文 context 回答问题。
- **CoT**：链式思维，先写推理步骤再给出 “Answer: …”。
- **ToT**：树式思维，多候选生成 → 按 F1 打分 → 保留最优 → 再生成/精化 → 再打分 → GroundTruth。
- **GoT**：图式思维，先将文档分成两组并生成两组摘要（Group 0 / Group 1）→ 对每组生成部分答案 → 聚合两步部分答案为最终答案 → 打分与 GroundTruth。

## 文件说明

- `utils.py`：加载 Hotpot JSON、context 转文本、答案归一化、F1/EM 与综合打分、GroundTruth 判定。
- `multi_hop_qa.py`：`MultiHopPrompter`、`MultiHopParser`、`io`/`cot`/`tot`/`got` 操作图、`run()` 入口。

## 运行方式

在项目根目录或本目录下执行（需已配置 `graph_of_thoughts/language_models/config.json` 及相应 API key）：

```bash
# 使用默认 ChatGPT 配置，跑 5 个样本、4 种方法
python -m examples.multi_hop_qa.multi_hop_qa

# 或指定语言模型名（需在 config.json 中存在）
python -m examples.multi_hop_qa.multi_hop_qa gemini-2.5-flash
```

在脚本内可修改：

- `samples`：要跑的样本下标列表。
- `approaches`：要跑的方法列表，如 `[io, cot, got]`。
- `budget`：费用上限（美元）。
- `data_path`：Hotpot JSON 路径；若为 `None`，则使用默认相对路径 `../../hotpot/hotpot_dev_distractor_v1.json`。

## 结果输出

- 在 `results/` 下生成以时间戳命名的子目录。
- 内含 `config.json`、`log.log`，以及每个方法名子目录下的各样本 GRS JSON（`< _id >.json`）。

## 使用 GCLI / Gemini

若要用 GCLI 代理或原生 Gemini，在 `multi_hop_qa.py` 的 `run()` 中构造 LM 时改用对应类，例如：

```python
from graph_of_thoughts.language_models.gcli_gemini import GCLIGemini
lm = GCLIGemini(config_lm_path, model_name="gemini-2.5-flash", cache=True)
```

并保证 `config.json` 中存在 `gemini-2.5-flash`（或所用模型名）的配置。
