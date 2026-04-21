# Makemore — Character-Level Language Models

A series of character-level language models for name generation, implemented in
PyTorch following Andrej Karpathy's "Building makemore" tutorial series.

Re-implemented from a working understanding of language modeling fundamentals
(not copy-paste) to solidify my knowledge of how autoregressive models work
from bigrams up to MLPs.

## Contents

- `Bigram.ipynb` — Bigram character-level model (counting + neural net approach)
- `MLP.ipynb` — MLP with character embeddings, trained on name data
- `MLP3.ipynb` — Extended MLP experiments (WIP)
- `names.txt` — Training dataset (32K names)
- `makemore.py` — Refactored model classes, training loop, and sampling utilities
- `test_makemore.py` — Tests for the models and training pipeline

## What the project covers

- Bigram language model via explicit counting and normalization
- Bigram model as a single-layer neural network with softmax
- Character embeddings learned jointly with the model
- MLP architecture: embedding → hidden layer (tanh) → output softmax
- Train/validation/test split for proper evaluation
- Mini-batch SGD with learning rate tuning
- Name generation via autoregressive sampling from the trained model
- Refactored `MakemoreModel` class with configurable architecture
- Training loop with loss tracking, learning rate scheduling, and early stopping
- Top-k and temperature-based sampling for controllable generation

## What I learned

- How bigram models bridge counting and gradient-based approaches
- Why embeddings let the model share statistical strength across similar characters
- The role of the hidden layer in capturing character interactions beyond bigrams
- How mini-batching stabilizes gradient estimates and speeds up training
- Why train/val/test splits matter even for generative models
- How temperature and top-k sampling trade diversity vs. quality in generation

## Next steps

- [ ] Add BatchNorm layer for more stable training
- [ ] Implement Wavenet-style residual connections
- [ ] Add a learning rate finder utility

## Credits

Based on [Andrej Karpathy's makemore tutorial](https://www.youtube.com/watch?v=PaCmpygF5o0)
and his [original implementation](https://github.com/karpathy/makemore). All educational
credit to him.
