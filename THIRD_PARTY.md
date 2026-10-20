# 来源与许可

本仓库的自编应用和脚本与第三方模型、数据分开保存。

| 来源 | 使用方式 | 参考 |
| --- | --- | --- |
| Qwen3.5-2B | 锁定提交下载基础模型，用于本机 LoRA | https://huggingface.co/Qwen/Qwen3.5-2B |
| CMRC2018 / Cui et al. | 中文抽取式阅读理解数据；下载后的分组和派生增强记录在 manifest | https://github.com/ymcui/cmrc2018 |
| hfl/cmrc2018 数据卡 | 声明 CC-BY-SA-4.0；派生数据再分发需遵守该许可 | https://huggingface.co/datasets/hfl/cmrc2018 |
| llama.cpp | 可选本地 GGUF 服务，MIT | https://github.com/ggml-org/llama.cpp |
| Unsloth Qwen3.5-4B GGUF | 可选基础模型量化转换，不是训练产物 | https://huggingface.co/unsloth/Qwen3.5-4B-GGUF |
| Qwen3 Embedding GGUF | 可选官方语义检索模型 | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF |

原始 CMRC 论文：Yiming Cui et al., *A Span-Extraction Dataset for Chinese Machine Reading Comprehension*, EMNLP-IJCNLP 2019，https://aclanthology.org/D19-1600/ 。

`data/demo/` 的星河实验室文档为项目原创虚构材料，与真实单位无关。训练数据、第三方二进制、模型权重默认不随代码提交；使用或再发布前应保留对应来源、许可证及模型卡要求。

## 本次 Git 发布范围

发布内容包括自编应用/训练/评测代码、原创虚构演示材料，以及 `evaluation/results/` 的统计与文件指纹。Qwen 基础权重、LoRA 适配器、tokenizer 文件、第三方二进制、CMRC2018 原始/派生记录、问题答案和逐条预测均不在本次发布中。统计摘要不授予对底层模型或数据的再分发权利。

上表中数据许可表述来自项目下载时保存的数据卡声明，原始来源信息同时保留在本地数据 manifest。模型与可选转换版本的实际使用及另行分发应以对应仓库的许可证和模型卡为准；项目代码的许可（如另行声明）不覆盖这些第三方资产。

## 独立部署附件

GitHub Release 另附训练所得 LoRA 部署包。Qwen3.5-2B 固定提交 `15852e8c16360a2fea060d615a32b45270f8a8fc` 的[官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/README.md)声明 Apache-2.0。部署包保留该提交的官方 LICENSE 和标明 LoRA 修改、数据来源及验证集选模规则的 NOTICE；官方许可证原文同时保存在 `docs/licenses/Qwen3.5-LICENSE.txt`。该附件不包含基础权重、tokenizer、训练记录或题目原文。
