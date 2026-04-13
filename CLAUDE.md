# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is Diana Vins' submission repository for the **Tactus Embodied AI Internship Take-Home Assessment**. It contains three questions (Q1, Q2, Q3), each in its own subdirectory.

## Assessment Structure

- **Q1/** — Written/visual explanation of why a physical task is hard for robots and how you'd teach it (flexible format: diagrams, paragraphs, bullets)
- **Q2/** — ML code: texture classifier trained on the [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) with 47 classes; must include code, test accuracy, a brief write-up, and at least one surprising finding
- **Q3/** — Open-ended: a project, script, paper reflection, failure lesson, or untested idea demonstrating curiosity about robots/AI

## Q2: ML Development

Q2 is the main coding task. Expected stack: **Python + PyTorch**.

### Dataset
```python
import torchvision.datasets as datasets
dtd = datasets.DTD(root='./data', split='train', download=True)
```
47 texture classes, ~5,600 images total.

### Likely commands once environment is set up
```bash
# Install dependencies (adapt to actual requirements.txt or pyproject.toml)
pip install torch torchvision

# Run training
python Q2/train.py

# Run evaluation
python Q2/evaluate.py
```

No build system or test runner exists yet — add commands here once Q2 code is written.

## Submission Notes

- LLMs are encouraged; evaluators will ask you to walk through answers — understanding the output matters
- Submission format: GitHub repo link (this repo), PDF, or Google Doc

## Code Style

Professional, clean and concise