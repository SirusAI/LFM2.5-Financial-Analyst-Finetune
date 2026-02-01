# LFM2.5 Fine-Tuning Project: Chinese Financial Analyst (LFM2.5 A股/CFA 金融分析微调项目)

## Overview | 项目概述
This project contains the complete workflow for fine-tuning the **Liquid LFM2.5-1.2B-Thinking** model into a specialized **Financial Analyst**. The resulting model is tailored for the Chinese market, specifically analyzing A-share stocks and applying CFA level financial expertise.

本项目提供了将 **Liquid LFM2.5-1.2B-Thinking** 模型微调为专业**金融分析师**的完整流程。微调后的模型专门针对中国A股市场个股分析、CFA专业财务知识应用进行了深度优化。

---

## Strategic Focus | 微调方向
1.  **Chinese A-Shares (A股个股分析)**: Training on massive financial research data to understand the unique dynamics and reporting style of the Chinese stock market. 使用大量行业研究报告进行训练，深入理解中国股市的独特动态和报告风格。
2.  **CFA Professional Standards (CFA专业标准)**: Integrating professional accounting principles and valuation methodologies sourced from CFA Level reasoning data. 整合了源自CFA核心课程的专业会计准则及估值方法论。
3.  **Thinking & Reasoning (思维推理)**: Leveraging the base model's "Thinking" architecture to perform logical chain-of-thought analysis on complex financial questions. 利用原模型的“思考”架构，对复杂的金融问题进行严密的逻辑链推演。

## Environment & Hardware | 环境与硬件
- **Hardware**: NVIDIA RTX 5070 Ti (16GB VRAM) - Blackwell Architecture.
- **Support**: Utilized **PyTorch Nightly (CUDA 12.8)** to enable support for the latest GPU architecture (`sm_120`).

## Project Structure | 项目结构
```
.
├── data/
│   └── merged_dataset.jsonl   # Comprehensive Financial Analysis Dataset
├── scripts/
│   ├── train.py               # HuggingFace LoRA Training Script
│   ├── test_model_simple.py   # Bilingual Verification Script
│   └── upload_to_hf.py        # Automation Script for HF Hub
├── model_output/              # Exported Safetensors & GGUF
└── README.md                  # This README
```

## How to Reproduce | 快速开始

### 1. Setup Environment
```bash
conda create -n finetune_env python=3.11 -y
conda activate finetune_env
# Install Nightly Torch for RTX 50-series
pip install --pre --upgrade torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers peft datasets trl gguf
```

### 2. Run Training
```bash
python scripts/train.py
```

### 3. Verification
```bash
python scripts/test_model_simple.py "分析某上市公司的财务健康状况。"
```

## Acknowledgements | 致谢
- **Liquid AI**: For the revolutionary LFM2.5 base model.
- **Hugging Face**: For the PEFT/TRL training ecosystem.
