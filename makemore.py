import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data(path="names.txt"):
    with open(path, "r") as f:
        words = f.read().splitlines()
    return words


def build_vocab(words):
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_dataset(words, stoi, block_size=3):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


def split_data(words, train_frac=0.8, val_frac=0.1, seed=42):
    random.seed(seed)
    shuffled = list(words)
    random.shuffle(shuffled)
    n = len(shuffled)
    n1 = int(train_frac * n)
    n2 = n1 + int(val_frac * n)
    return shuffled[:n1], shuffled[n1:n2], shuffled[n2:]


class MakemoreModel(nn.Module):
    def __init__(self, vocab_size, n_emb=10, n_hidden=300, block_size=3, seed=2147483647):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        g = torch.Generator().manual_seed(seed)
        self.C = nn.Parameter(torch.randn((vocab_size, n_emb), generator=g))
        self.W1 = nn.Parameter(torch.randn((n_emb * block_size, n_hidden), generator=g))
        self.b1 = nn.Parameter(torch.randn(n_hidden, generator=g))
        self.W2 = nn.Parameter(torch.randn((n_hidden, vocab_size), generator=g))
        self.b2 = nn.Parameter(torch.randn(vocab_size, generator=g))

    def forward(self, X):
        emb = self.C[X]
        emb_cat = emb.view(emb.shape[0], -1)
        h = torch.tanh(emb_cat @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def loss(self, logits, Y):
        return F.cross_entropy(logits, Y)


class BigramModel(nn.Module):
    def __init__(self, vocab_size, seed=2147483647):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn((vocab_size, vocab_size), generator=g))

    def forward(self, X):
        xenc = F.one_hot(X, num_classes=self.W.shape[0]).float()
        logits = xenc @ self.W
        return logits

    def loss(self, logits, Y):
        return F.cross_entropy(logits, Y)


class Trainer:
    def __init__(self, model, X_tr, Y_tr, X_val=None, Y_val=None,
                 lr=0.1, batch_size=32, seed=42, verbose=True):
        self.model = model
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.X_val = X_val
        self.Y_val = Y_val
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.train_losses = []
        self.val_losses = []

    def _eval_loss(self, X, Y, max_samples=1000):
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            n = min(X.shape[0], max_samples)
            logits = self.model(X[:n])
            loss = self.model.loss(logits, Y[:n])
        if was_training:
            self.model.train()
        return loss.item()

    def fit(self, max_steps=20000, lr_decay=None, early_stop_patience=None):
        g = torch.Generator().manual_seed(self.seed)
        best_val_loss = float("inf")
        patience_counter = 0

        for step in range(max_steps):
            ix = torch.randint(0, self.X_tr.shape[0], (self.batch_size,), generator=g)
            logits = self.model(self.X_tr[ix])
            loss = self.model.loss(logits, self.Y_tr[ix])

            self.model.zero_grad()
            loss.backward()

            lr = self.lr
            if lr_decay is not None:
                lr = lr_decay(step, self.lr)

            for p in self.model.parameters():
                p.data -= lr * p.grad

            if step % 1000 == 0:
                tr_loss = loss.item()
                self.train_losses.append(tr_loss)
                msg = f"Step {step:5d} | train loss {tr_loss:.4f}"
                if self.X_val is not None:
                    val_loss = self._eval_loss(self.X_val, self.Y_val)
                    self.val_losses.append(val_loss)
                    msg += f" | val loss {val_loss:.4f}"

                    if early_stop_patience is not None:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stop_patience:
                                if self.verbose:
                                    print(f"Early stopping at step {step}")
                                break
                if self.verbose:
                    print(msg)

        if self.verbose:
            final_tr = self._eval_loss(self.X_tr, self.Y_tr)
            msg = f"Final | train loss {final_tr:.4f}"
            if self.X_val is not None:
                final_val = self._eval_loss(self.X_val, self.Y_val)
                msg += f" | val loss {final_val:.4f}"
            print(msg)


def cosine_lr(step, base_lr, warmup=1000, total=20000):
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / (total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def sample(model, itos, n_samples=20, block_size=3, seed=2147483647,
           temperature=1.0, top_k=None):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    names = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = []
            context = [0] * block_size
            while True:
                x = torch.tensor([context])
                logits = model(x)
                logits = logits / temperature

                if top_k is not None and top_k < logits.shape[-1]:
                    values, _ = torch.topk(logits, top_k)
                    min_val = values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_val,
                                         torch.full_like(logits, float("-inf")),
                                         logits)

                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
            names.append("".join(itos[i] for i in out[:-1]))
    model.train()
    return names


import math


if __name__ == "__main__":
    words = load_data("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)
    block_size = 3

    train_w, val_w, test_w = split_data(words)
    X_tr, Y_tr = build_dataset(train_w, stoi, block_size)
    X_val, Y_val = build_dataset(val_w, stoi, block_size)
    X_te, Y_te = build_dataset(test_w, stoi, block_size)

    print(f"Train: {X_tr.shape[0]} | Val: {X_val.shape[0]} | Test: {X_te.shape[0]}")

    model = MakemoreModel(vocab_size, n_emb=10, n_hidden=300, block_size=block_size)
    trainer = Trainer(model, X_tr, Y_tr, X_val, Y_val, lr=0.1, batch_size=32)
    trainer.fit(max_steps=20000, lr_decay=cosine_lr, early_stop_patience=5)

    print("\n=== Sampling (temperature=0.8, top_k=10) ===")
    names = sample(model, itos, n_samples=10, block_size=block_size,
                   temperature=0.8, top_k=10, seed=42)
    for name in names:
        print(name)

    print("\n=== Sampling (temperature=1.0, no top_k) ===")
    names = sample(model, itos, n_samples=10, block_size=block_size,
                   temperature=1.0, seed=42)
    for name in names:
        print(name)
