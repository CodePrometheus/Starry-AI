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

- Use `uv` to run Python scripts

## Learning Principles

- This repository is for hands-on, code-first learning rather than abstract discussion detached from implementation
- Learning must follow the predefined roadmap without drifting, skipping around, or expanding without clear scope
- The default learning style should be: concept breakdown, classic examples, manual implementation, and run-time verification
- Prefer understanding the principles through handwritten code and step-by-step code decomposition before moving to framework abstractions
- When explaining abstract code, formulas, or data transformations, use the actual data in the current script to give at least one concrete worked example before generalizing
- Use the current script's real samples, feature values, matrices, and outputs as the default teaching examples; avoid placeholder or fabricated examples when real data is available
- Code comments should optimize for learner understanding: explain what a specific real input becomes after the transformation, not just abstract shapes or API behavior
- Avoid outline-style or slogan-style explanations in learning materials when the user is asking for direct concept clarification
- When multiple columns or cases follow the same rule, explain the general rule once and then show one representative worked example instead of repeating the same explanation mechanically
- Learning scripts may use `TODO(human)` to mark exercises that should be completed by the user
- Once the user has implemented the exercise correctly, the corresponding `TODO(human)` should be removed to mark it as completed

## Commit Requirements

- Every git commit message must include a `Co-authored-by` trailer
- The `Co-authored-by` value must reflect the actual model or coding assistant used in that specific change
- Example values include: `Co-authored-by: GitHub Copilot`, `Co-authored-by: Claude`, and `Co-authored-by: gemini-code-assist`
