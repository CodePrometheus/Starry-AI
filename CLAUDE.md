# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal AI/ML learning repository with standalone Python tutorial scripts covering PyTorch, computer vision, and LLM application development. MIT licensed.

## Running Code

No build system or test suite. Each `.py` file runs independently:

```bash
pip install -r requirements.txt
python <path/to/script.py>
```

The QA bot API (the most structured component):
```bash
python langchain_learning/examples/langchain_bot/api.py  # FastAPI on port 8100
```

## Architecture

Four independent module areas:

- **`tensor/`** — PyTorch tensor tutorials (creation, indexing, PCA, LDA, image ops)
- **`neuralnetwork/`** — Neural networks from scratch and with PyTorch (backprop with numpy, CNN, regression, MNIST/housing classification). Models saved as `.pkl` in `neuralnetwork/model/`
- **`cifar/`** — CIFAR-10 classification with custom dataset loading, augmentation, and VGGNet architecture
- **`langchain_learning/`** — LangChain tutorials (LLMs, chains, agents, vector stores) plus a complete document QA bot in `examples/langchain_bot/` using FAISS, FastAPI, and Gradio

## Key Patterns

- PyTorch `nn.Module` subclassing for all model definitions
- Standard PyTorch training loops (forward → loss → backward → optimizer step)
- LangChain patterns: prompt templates, RetrievalQA/LLMChain/SQLDatabaseChain, agents with tools
- `langchain_bot/` uses a `LocalDocQA` class wrapping FAISS vector store, exposed via FastAPI REST endpoints

## Key Dependencies

torch, torchvision, numpy, langchain, fastapi, uvicorn, gradio, Pillow, opencv-python, matplotlib, pandas, nltk

## Code Style Preferences

- 注释要丰富详细，尤其是学习脚本中的数学公式、计算过程和直觉解释，帮助理解每一步在做什么
- 运行 Python 脚本使用：`/opt/homebrew/bin/uv run /Users/zhouzixin/Learn/mini-aime/.venv/bin/python3 <script.py>`
