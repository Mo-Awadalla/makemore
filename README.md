# makemore — Character-Level Language Modeling Toolkit

A from-scratch PyTorch toolkit for training and sampling character-level language models, progressing from bigram counting to MLPs with learned embeddings. Built to understand autoregressive sequence modeling at the implementation level.

## Overview

This project implements a complete training and inference pipeline for name generation models, with modular components for data handling, model architectures, training loops, and controllable sampling.

## What's Implemented

### Data Pipeline
- Vocabulary construction from raw text corpora
- Sliding-window dataset building with configurable context lengths
- Deterministic train/validation/test splits

### Model Architectures
- **BigramModel**: Single-layer neural net formulation of bigram counting, trained with gradient descent instead of explicit frequency tables
- **MakemoreModel**: MLP with learned character embeddings, hidden tanh layer, and output softmax; supports configurable embedding dimensions, hidden size, and context window

### Training Infrastructure
- `Trainer` class with mini-batch SGD, learning rate scheduling, and early stopping
- Cosine learning rate decay with linear warmup
- Train/validation loss tracking and logging
- Deterministic seeding for reproducibility

### Sampling & Generation
- Autoregressive character-by-character generation
- Temperature-based sampling for diversity control
- Top-k filtering for quality/concentration tradeoff
- Deterministic sampling for reproducible outputs

### Testing
Full pytest suite covering data utilities, model forward passes, gradient flow, training convergence, validation splitting, early stopping, and sampling determinism.

## Quick Start

```bash
pip install torch pytest
python makemore.py
```

This trains an MLP on the included 32,000-name dataset and samples names with temperature and top-k filtering.

## Project Structure

| File | Description |
|------|-------------|
| `makemore.py` | Core library: data utils, model classes, trainer, sampling |
| `test_makemore.py` | pytest suite for all components |
| `Bigram.ipynb` | Bigram model exploration (counting + neural net) |
| `MLP.ipynb` | MLP with embeddings: training, evaluation, generation |
| `names.txt` | Training corpus: 32K names |

## Key Design Decisions

- **Modular architecture**: `MakemoreModel`, `BigramModel`, `Trainer`, and `sample()` are decoupled so any model can be trained with the same trainer and sampled with the same generator
- **Configurable generation**: Temperature and top-k are passed at inference time, not hardcoded, making it easy to experiment with output quality
- **Testable training**: The `Trainer` class exposes `train_losses` and `val_losses` arrays so experiments can be validated programmatically

## License

MIT
