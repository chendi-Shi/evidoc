# Evidoc · 见微

**在 RTX 3060 Ti 8GB 上完成中文证据问答 LoRA 微调，并将适配器接入本地文档助手。**

项目覆盖文档导入与检索、数据分组划分、真实 GPU 训练、完整断点恢复、三种配置 × 三个随机种子的实验、基础模型对照评测，以及本地服务。正式实验的 **9 组训练与 9 组成对评测已于 2026-10-16 完成**；96 小时是预算上限，实际训练调度耗时约 **26.54 小时**，各组均正常触发验证早停。

[实验报告](docs/EXPERIMENTS.md) · [复现步骤](docs/REPRODUCING.md) · [机器可读结果](evaluation/results/experiment_summary.json) · [逐次指标 CSV](evaluation/results/per_run_metrics.csv) · [第三方来源](THIRD_PARTY.md)

2026-10-20 已完成实际 CUDA 端到端验收：上传虚构文档、BM25 检索、训练所得 LoRA 回答、逐字出处校验，以及空库拒答、模型离线回退、文档删除。三个短答案用例全部通过，实测服务调用约 1.6～4.6 秒；这是集成冒烟测试，不是用户文档准确率评测。[验收记录](evaluation/results/delivery_smoke.json)

![实际 LoRA 文档问答页面](docs/images/document-qa.jpg)

## 实测结果

所有模型使用相同的 **200 道留出题**，贪心解码、BF16、最长完整样本 512 token。下表为三个训练随机种子的均值 ± 样本标准差，单位为百分比；EM 和 F1 为本项目的自定义归一化字符指标，**不是 CMRC2018 官方榜单成绩**。

| 配置 | 可训练参数 | EM (%) | 字符 F1 (%) |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B 基础模型 | — | 61.00 | 84.80 |
| LoRA r8，混合架构投影 | 3,371,520 | 66.00 ± 1.32 | 88.89 ± 0.50 |
| LoRA r16，混合架构投影 | 6,743,040 | **68.83 ± 1.76** | **90.12 ± 0.45** |
| LoRA r8，全注意力 q/v | 417,792 | 66.83 ± 1.04 | 89.20 ± 0.86 |

r16 配置的平均 EM 提高 **7.83 个百分点**，F1 提高 **5.32 个百分点**。这是固定样本上的实验结果，不能直接推断用户上传文档的端到端准确率，也未进行显著性检验。

**部署选择仅依据验证损失**：九组中验证损失最低的是 `r16-hybrid-s43`，其最佳检查点为第 400 步，最终保存的 `adapter/` 来自该最佳检查点。该单次模型的留出题 EM 为 67.00%、F1 为 89.71%；不要把三次均值当作部署模型的单次分数，也不要用留出题最高分挑选部署种子。

## 项目组成

| 部分 | 实现 |
| --- | --- |
| 文档助手 | FastAPI、SQLite、原生 HTML/CSS/JS；导入 PDF / DOCX / Markdown / TXT |
| 检索 | 中文 bigram BM25、父子分块；可选语义向量与 RRF 混合检索 |
| 微调 | Qwen3.5-2B BF16 LoRA，冻结基础权重、梯度检查点、答案部分监督 |
| 实验 | 3 种适配配置 × seeds 42/43/44；每 50 步验证与保存，验证早停 |
| 恢复 | 优化器、调度器、RNG、早停状态与累计预算恢复；断点完整性检查和进程互斥 |
| 评测 | 同题同提示的基础模型/适配器对照；EM/F1、受控空上下文探针 |
| 部署 | `scripts.serve_adapter` 在 GPU 加载实际 2B LoRA；网页显示来源与生成状态 |
| 发布 | 只提交代码、原创演示资料和脱敏统计结果；CPU CI 无需模型或训练数据 |

这里的“混合架构投影”指 Qwen 的全注意力与 Gated DeltaNet 投影模块；“混合检索”指 BM25 与向量召回，是两个不同概念。选用 2B 模型是为了满足 8GB 单卡约束，不宣称全球 SOTA。

## 本地启动

Python 3.12，先安装应用依赖即可演示文档导入、检索与引用。无模型服务时，界面明确显示“原文摘录”。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.api:app --host 127.0.0.1 --port 8767
```

打开 [本地助手](http://127.0.0.1:8767)，导入内置的星河实验室虚构材料，再提问。API 文档位于 `/api/docs`。已有项目隔离运行时的 Windows 环境也可使用 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1`。

### 使用训练所得适配器

**直接演示无需重训**：[v0.1.0 发布附件](https://github.com/chendi-Shi/evidoc/releases/tag/v0.1.0)提供约 23.8 MiB 的 `evidoc-adapter.zip`。它只包含选中 LoRA、部署指纹与许可证，不包含基础模型或训练数据；文件 SHA-256 见[部署包清单](evaluation/results/deployment_bundle.json)。私有仓库需要先登录 GitHub 才能下载附件。

在安装了 GPU 训练依赖的 Python 环境中下载固定基础模型，将附件保存到项目目录后执行：

```powershell
python -m scripts.download_training_model --size 2B --mirror modelscope --connections 8
Expand-Archive -LiteralPath .\evidoc-adapter.zip -DestinationPath .\models\evidoc-adapter
python -m scripts.serve_adapter --bundle models/evidoc-adapter --inspect
python -m scripts.serve_adapter --bundle models/evidoc-adapter --port 8083
```

服务核验基础模型、tokenizer、LoRA 和部署清单，再加载 CUDA；不需要优化器或原始训练数据。完成[复现流程](docs/REPRODUCING.md)后，也可直接使用本机正式实验产物：

```powershell
python -m scripts.serve_adapter --experiment runs/experiment-96h --port 8083
```

保持服务运行，在另一个终端启动上面的网页服务。默认文档问答连接这个适配器服务。服务从已完成实验中按验证损失选择适配器，检查模型与适配器身份后加载；不要同时启动占用同一张 8GB 显卡的训练或其他大模型。

可选 GGUF 基础模型服务使用 `scripts/download_models.py` 和 `scripts/run_models.py`。必须显式设置 `EVIDOC_GENERATION_BACKEND=gguf` 才使用它；GGUF 4B 基线的页面效果不属于本次 2B LoRA 微调结果。

## 复现实验与断点恢复

详细步骤见 [REPRODUCING.md](docs/REPRODUCING.md)。主要入口如下：

```powershell
python -m scripts.download_training_model --size 2B --mirror modelscope --connections 8
python -m training.prepare_data --source hf-parquet --revision 137f2c45a24275fb68f6961c4d357f46288886aa
python -m training.train --model models/Qwen3.5-2B --output runs/pilot-2b-512 --pilot --pilot-steps 5 --max-length 512 --accumulation 16 --logging-steps 1 --save-steps 5
python -m scripts.run_experiments --model models/Qwen3.5-2B --pilot runs/pilot-2b-512/pilot_report.json --output runs/experiment-96h --hours 96 --max-length 512 --accumulation 16 --seeds 42,43,44 --evaluate-after
```

训练入口要求 CUDA；不会静默退回 CPU。下载源需通过锁定的官方提交与 SHA-256 校验。超长样本整条过滤；标签只覆盖助手答案和结束符。

下面的一键控制入口仅支持 `scripts/setup.ps1` 创建的 `.tools/python/` Windows 隔离环境：

```powershell
.\.tools\python\python.exe -m scripts.manage_training --action status
powershell -NoProfile -ExecutionPolicy Bypass -File .\pause-training.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\resume-training.ps1
```

每 50 个优化器步保存完整断点；暂停请求在当前优化器步结束后保存。恢复会跳过已完成实验，验证文件结构与 SHA-256 提交清单，损坏的最新断点会回退到较早的完整断点。参数与累计预算沿用原实验；已存活的进程不会被重复启动。这是手动恢复入口，不是开机自启服务。

普通 venv 环境不使用上述控制脚本；在原 `scripts.run_experiments` 命令末尾添加 `--resume` 继续实验，其他参数和输出目录保持一致，详见复现说明。

正式训练与最终评测完成后，用 CPU 校验本地证据并生成可发布结果：

```powershell
python -m scripts.export_results
python -m scripts.export_results --check
```

导出器核对 9 组状态、同一批样本、原始预测文件与适配器哈希，并从预测重算指标。导出内容只有统计、配置、时间、依赖版本和指纹；不包含题目、正文、答案、预测文本、样本 ID、绝对本机路径或进程信息。

## 测试与边界

```powershell
python -m pytest -q
```

GitHub Actions 在 Windows / Linux 的 Python 3.12 上运行 CPU 单测，不安装 PyTorch、不下载模型。单测覆盖检索、引用验证、数据划分、指标、断点损坏回退、恢复互斥及发布结果隐私边界；真实 GPU 训练和最终评测的统计证据保存在 `evaluation/results/`。

启动 GPU 适配器服务后，可用 `python -m scripts.verify_delivery` 重跑端到端验收。它使用临时文档库，不修改用户已有资料；生成的记录只有内置虚构问题与回答。推理窗口最多覆盖前三个来源、每来源两个窗口；超出范围会提示，短答案的逐字出处通过也不等于问题的实体、时间和逻辑均被正确理解。

- 数据来自 CMRC2018，按源文档和归一化上下文分组；原始开发集用作留出测试。无法排除基础模型预训练已见过公开语料，也未保证消除语义近重复。
- 留出集 3219 条中，831 条因完整样本超过 512 token 被过滤；从剩余 2388 条的固定顺序选 200 条评测，结论受样本量和长度筛选限制。
- 空上下文探针共 50 条，基础模型拒答率 98%、所有适配器 100%。它只反映受控空上下文策略，不等于真实无答案识别能力。
- 内置 12 道虚构制度题仅用于检索冒烟测试，Recall/MRR 不能作为生成问答准确率。
- 引用编号或摘录存在不证明整段回答正确。PDF 只提取已有文本，不含 OCR；DOCX 不推算 Word 排版页码。
- 文档助手默认仅监听本机地址，面向单用户本地演示；没有多租户权限与公网部署加固。

## 目录与发布范围

`app/` 文档服务；`static/` 界面；`training/` 训练与评测；`scripts/` 下载、服务、调度与结果导出；`tests/` 单测；`docs/` 实验说明；`evaluation/results/` 可公开统计。

`.gitignore` 排除模型权重、`runs/`、`reports/`、`.tools/`、运行日志、真实训练数据及用户数据库，保留 `models/manifest.json` 源配置。仓库只带原创虚构演示资料；模型、适配器、第三方原始/派生数据仍留在本地，其来源和再分发约束见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 简历表述示例

> 在 RTX 3060 Ti 8GB 上完成 Qwen3.5-2B 的 9 组 LoRA 消融实验，构建按文档分组的数据划分、答案监督、验证早停、完整断点恢复及模型指纹校验。在固定 200 道留出题上，r16 配置三种子平均 EM 从 61.00% 提高至 68.83%，字符 F1 从 84.80% 提高至 90.12%（自定义指标）；按验证损失选择适配器接入带来源引用的本地文档助手，并提供 CPU CI 与脱敏实验报告。
