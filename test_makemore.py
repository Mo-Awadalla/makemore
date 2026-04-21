import pytest
import torch
import os
from makemore import (
    load_data, build_vocab, build_dataset, split_data,
    MakemoreModel, BigramModel, Trainer, sample, cosine_lr
)


@pytest.fixture
def data():
    path = os.path.join(os.path.dirname(__file__), "names.txt")
    if not os.path.exists(path):
        pytest.skip("names.txt not found")
    words = load_data(path)
    stoi, itos = build_vocab(words)
    return words, stoi, itos


class TestDataUtils:
    def test_load_data(self, data):
        words, _, _ = data
        assert len(words) > 0
        assert all(isinstance(w, str) for w in words)

    def test_build_vocab(self, data):
        words, stoi, itos = data
        assert stoi["."] == 0
        assert len(stoi) == len(itos)
        for s, i in stoi.items():
            assert itos[i] == s

    def test_build_dataset(self, data):
        words, stoi, _ = data
        X, Y = build_dataset(words[:5], stoi, block_size=3)
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == 3
        assert X.dtype == torch.int64
        assert Y.dtype == torch.int64

    def test_build_dataset_block_size(self, data):
        words, stoi, _ = data
        X, Y = build_dataset(words[:5], stoi, block_size=5)
        assert X.shape[1] == 5

    def test_split_data(self, data):
        words, _, _ = data
        train, val, test = split_data(words)
        total = len(train) + len(val) + len(test)
        assert total == len(words)
        assert len(train) >= len(val)
        assert len(train) > 0 and len(val) > 0 and len(test) > 0


class TestMakemoreModel:
    def test_init(self, data):
        _, stoi, _ = data
        model = MakemoreModel(len(stoi), n_emb=10, n_hidden=100, block_size=3)
        params = list(model.parameters())
        assert len(params) == 5

    def test_forward(self, data):
        words, stoi, _ = data
        model = MakemoreModel(len(stoi), n_emb=10, n_hidden=100, block_size=3)
        X, Y = build_dataset(words[:3], stoi, block_size=3)
        logits = model(X)
        assert logits.shape == (X.shape[0], len(stoi))

    def test_loss(self, data):
        words, stoi, _ = data
        model = MakemoreModel(len(stoi), n_emb=10, n_hidden=100, block_size=3)
        X, Y = build_dataset(words[:3], stoi, block_size=3)
        logits = model(X)
        loss = model.loss(logits, Y)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_backward(self, data):
        words, stoi, _ = data
        model = MakemoreModel(len(stoi), n_emb=10, n_hidden=100, block_size=3)
        X, Y = build_dataset(words[:3], stoi, block_size=3)
        logits = model(X)
        loss = model.loss(logits, Y)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestBigramModel:
    def test_init(self, data):
        _, stoi, _ = data
        model = BigramModel(len(stoi))
        assert model.W.shape == (len(stoi), len(stoi))

    def test_forward(self, data):
        words, stoi, _ = data
        model = BigramModel(len(stoi))
        X = torch.tensor([0, 5, 13])
        logits = model(X)
        assert logits.shape == (3, len(stoi))

    def test_loss(self, data):
        words, stoi, _ = data
        model = BigramModel(len(stoi))
        X = torch.tensor([0, 5, 13])
        Y = torch.tensor([5, 13, 1])
        logits = model(X)
        loss = model.loss(logits, Y)
        assert loss.item() > 0


class TestTrainer:
    def test_fit_reduces_loss(self, data):
        words, stoi, _ = data
        X, Y = build_dataset(words[:50], stoi, block_size=3)
        model = MakemoreModel(len(stoi), n_emb=5, n_hidden=50, block_size=3)
        trainer = Trainer(model, X, Y, lr=0.1, batch_size=16, verbose=False)
        initial_loss = trainer._eval_loss(X, Y)
        trainer.fit(max_steps=500)
        final_loss = trainer._eval_loss(X, Y)
        assert final_loss < initial_loss

    def test_fit_with_validation(self, data):
        words, stoi, _ = data
        train_w, val_w, _ = split_data(words[:100])
        X_tr, Y_tr = build_dataset(train_w, stoi, block_size=3)
        X_val, Y_val = build_dataset(val_w, stoi, block_size=3)
        model = MakemoreModel(len(stoi), n_emb=5, n_hidden=50, block_size=3)
        trainer = Trainer(model, X_tr, Y_tr, X_val, Y_val, lr=0.1, batch_size=16, verbose=False)
        trainer.fit(max_steps=500)
        assert len(trainer.train_losses) > 0
        assert len(trainer.val_losses) > 0

    def test_early_stopping(self, data):
        words, stoi, _ = data
        train_w, val_w, _ = split_data(words[:100])
        X_tr, Y_tr = build_dataset(train_w, stoi, block_size=3)
        X_val, Y_val = build_dataset(val_w, stoi, block_size=3)
        model = MakemoreModel(len(stoi), n_emb=5, n_hidden=50, block_size=3)
        trainer = Trainer(model, X_tr, Y_tr, X_val, Y_val, lr=0.1, batch_size=16, verbose=False)
        trainer.fit(max_steps=50000, early_stop_patience=3)
        assert trainer.train_losses[-1] > 0

    def test_cosine_lr(self):
        assert cosine_lr(0, 0.1) == pytest.approx(0.0, abs=1e-6)
        assert cosine_lr(500, 0.1) > 0
        assert cosine_lr(20000, 0.1) >= 0
        assert cosine_lr(20000, 0.1) <= 0.1


class TestSampling:
    def test_sample_basic(self, data):
        _, stoi, itos = data
        model = MakemoreModel(len(stoi), n_emb=10, n_hidden=100, block_size=3)
        names = sample(model, itos, n_samples=5, block_size=3, seed=42)
        assert len(names) == 5
        assert all(isinstance(n, str) for n in names)

    def test_sample_with_temperature(self, data):
        _, stoi, itos = data
        model = MakemoreModel(len(stoi), n_emb=10, n_hidden=100, block_size=3)
        names_low = sample(model, itos, n_samples=3, block_size=3, temperature=0.5, seed=42)
        names_high = sample(model, itos, n_samples=3, block_size=3, temperature=2.0, seed=42)
        assert len(names_low) == 3
        assert len(names_high) == 3

    def test_sample_with_top_k(self, data):
        _, stoi, itos = data
        model = MakemoreModel(len(stoi), n_emb=10, n_hidden=100, block_size=3)
        names = sample(model, itos, n_samples=3, block_size=3, top_k=5, seed=42)
        assert len(names) == 3

    def test_sample_deterministic(self, data):
        _, stoi, itos = data
        model = MakemoreModel(len(stoi), n_emb=10, n_hidden=100, block_size=3)
        names1 = sample(model, itos, n_samples=3, block_size=3, seed=123)
        names2 = sample(model, itos, n_samples=3, block_size=3, seed=123)
        assert names1 == names2
