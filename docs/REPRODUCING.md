# 从源码复现实验

本仓库的公开结果来自 2026-10-15 至 2026-10-16 的本机实验。复现需要自行下载权重、数据并训练；Git 克隆不会附带几 GB 的模型、训练数据库或 LoRA 权重。相同配置不保证跨硬件、库版本和非确定性算子得到逐 bit 相同结果。

## 1. 环境

实测环境：Windows 10、RTX 3060 Ti 8192 MiB、32GB RAM、i7-11700；基础模型在 BF16、未量化模式运行，PyTorch 2.7.1+cu118 / CUDA 11.8。驱动 472.80 在本机通过前向、反向和优化器探测；这只是实测记录，不是其他机器的兼容承诺。

Windows 可运行项目隔离安装脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Training
$python = '.\.tools\python\python.exe'
$env:PYTHONUTF8 = '1'
```

网络需要时可向 setup 添加 `-UseChinaMirror`。脚本使用项目下的隔离 Python，不改系统 Python 包。一键暂停与恢复脚本也绑定这个 `.tools/python/` 环境。

普通 Python 3.12 venv 也可安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r training/requirements.txt
$python = 'python'
$env:PYTHONUTF8 = '1'
```

在 Linux 上使用相应的 venv 激活命令。已完成的 GPU 实测来自 Windows，CI 的 Linux 覆盖仅限 CPU 单测。后续命令中的 `& $python` 是 PowerShell 调用语法，在其他 shell 中替换为相应 Python 命令。

## 2. 固定权重和数据

```powershell
& $python scripts/probe_training_gpu.py
& $python -m scripts.download_training_model --size 2B --mirror modelscope --connections 8
& $python -m training.prepare_data --source hf-parquet --revision 137f2c45a24275fb68f6961c4d357f46288886aa
```

模型锁定官方 `Qwen/Qwen3.5-2B` 提交 `15852e8c16360a2fea060d615a32b45270f8a8fc`，权重文件 SHA-256：

```text
aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1
```

ModelScope 在这条命令中仅提供下载传输，文件必须通过官方长度与哈希校验。数据锁定 `hfl/cmrc2018` 提交，原始 parquet 文件哈希、派生分组数量与分区哈希见 [公开统计](../evaluation/results/experiment_summary.json)。`manifest.json` 已存在时，数据准备入口不会覆盖它；需要新划分时指定新输出目录。

公开报告中的 `dataset_manifest_sha256` 包含本次准备时间等元数据，新机器重新生成 manifest 后不必相同。应核对固定源提交、原始文件 SHA-256、各分区内容 SHA-256 和数量，而不是修改结果来追求相同 manifest 时间戳。

## 3. 先做 GPU 短跑

```powershell
& $python -m training.train --model models/Qwen3.5-2B --output runs/pilot-2b-512 --pilot --pilot-steps 5 --max-length 512 --accumulation 16 --logging-steps 1 --save-steps 5
```

确认 `pilot_report.json` 与 GPU 读数正常后再开始正式训练。短跑的 5 步中位耗时曾为 19.84 秒，不能据此承诺最终完成时间。当前正式实验每组最多 1356 步，但早停会减少步数。显存不足会报错并退出，不会静默改为 CPU 训练。

## 4. 正式九组与最终评测

```powershell
& $python -m scripts.run_experiments --model models/Qwen3.5-2B --pilot runs/pilot-2b-512/pilot_report.json --output runs/experiment-96h --hours 96 --max-length 512 --accumulation 16 --seeds 42,43,44 --evaluate-after
```

同一张 8GB GPU 上不要并行加载推理模型。运行器顺序执行三种配置 × 三个随机种子；每组最多三个 epoch，并依据固定验证子集早停。`--hours 96` 是累计预算上限，不用于填满空闲时长。

每次训练结束会恢复该运行的最佳验证权重，再保存 `adapter/`。`--evaluate-after` 随后评测基础模型和适配器，使用固定 200 道留出题与 50 条人工空上下文探针。最终报告位于 `runs/experiment-96h/final_comparison.json`；只有训练和评测都完成，才算这套实验完成。

本地中间产物包括：`run_manifest.json`、`metrics.jsonl`、`tokenization_report.json`、`checkpoint-*/`、`adapter/` 和每组的 `heldout-evaluation/`。这些目录包含模型、数据派生信息和绝对路径，不直接提交到 Git。

## 5. 暂停、恢复与检查

使用项目内置 Python 的 Windows 环境：

```powershell
& $python -m scripts.manage_training --action status
powershell -NoProfile -ExecutionPolicy Bypass -File .\pause-training.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\resume-training.ps1 -WhatIf
powershell -NoProfile -ExecutionPolicy Bypass -File .\resume-training.ps1
```

暂停请求在当前优化器步完成后保存。断点包含 LoRA 权重、优化器、调度器、RNG、步数与早停计数；提交清单使用文件大小和 SHA-256 识别不完整写入。最新断点损坏时，会选择之前的完整断点并保留损坏文件。回退造成需要重放的历史指标归档到 `recovery-history/`，已用时间仍计入预算。

普通 venv 环境恢复时，在第 4 节的完整命令末尾加 `--resume`。参数、种子和输出目录必须与原实验一致。需要暂停时可创建 `runs/experiment-96h/STOP_REQUESTED`；运行器将在当前步骤完成并保存后停止。`--resume` 会处理该停止标记；不用固定路径的 `manage_training` 包装入口。

进程互斥可防止重复写同一实验目录。恢复脚本是显式手动恢复，不是系统开机服务，也不会无限重试失败任务。

## 6. 校验并导出公开结果

```powershell
& $python -m scripts.export_results
& $python -m scripts.export_results --check
```

默认读取本地 `runs/experiment-96h` 并写入：

- `evaluation/results/experiment_summary.json`：配置、三种子统计、逐次训练步骤、验证选择、模型/数据/预测文件哈希及必要运行环境。
- `evaluation/results/per_run_metrics.csv`：便于独立复算的九次运行统计。

导出器要求全部运行和最终评测完整；核对 adapter 与预测文件哈希、固定测试样本顺序和评测设置，并使用已有评分函数从原始预测重新计算 EM/F1/拒答指标。输出采用明确的字段清单，不含题目、正文、答案、预测文本、样本 ID、绝对本机路径和进程信息。重复执行 `--check` 可验证公开文件与本地证据仍一致；没有本地实验产物的源码克隆不能执行这项证据检查，但可以直接阅读公开统计。

## 7. 部署实际 LoRA

```powershell
& $python -m scripts.serve_adapter --experiment runs/experiment-96h --port 8083
```

在另一个终端运行 `python -m uvicorn app.api:app --host 127.0.0.1 --port 8767`。默认服务按完成运行的最低验证损失选择适配器，本次为 `r16-hybrid-s43`，权重来自第 400 步最佳验证检查点。公开测试分数只用于结果报告，不参与服务选择。

模型权重未发布时，克隆项目可以直接演示检索和原文摘录；需要模型回答则先复现训练，或在已完成实验的本机启动上述服务。可选 GGUF 基线通过 `EVIDOC_GENERATION_BACKEND=gguf` 明确选择，其结果不属于 2B LoRA 实验。

## 8. CPU 测试

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

测试不需要 CUDA、PyTorch、联网模型或真实训练语料；GitHub Actions 在 Windows 和 Linux 的 Python 3.12 上执行。GPU 操作的有效性由实际训练、评测产物和独立运行探测提供证据，不能用 CPU CI 通过来代替 GPU 成功。
