"""
apps/train_lm_lstm.py
======================
Train an LSTM language model on Penn Treebank.

Usage:
  python apps/train_lm_lstm.py --data data/ptb \\
                                --epochs 20 --lr 0.003 \\
                                --hidden_size 256 --seq_len 35

No PyTorch / TensorFlow — pure MiniGrad.
"""
import argparse
import sys
import os
import time
import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import minigrad.nn as nn
import minigrad.optim as optim
from minigrad.autograd import Tensor
from minigrad.data import Corpus, batchify, get_batch
from apps.models import LanguageModel


def evaluate_ppl(model, data, seq_len):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.SoftmaxLoss()
    h = None
    for i in range(0, data.shape[0] - 1, seq_len):
        x, y = get_batch(data, i, seq_len)
        logits, h = model(x, h)
        if isinstance(h, tuple):
            # Detach hidden state to prevent BPTT through full sequence
            h = tuple(Tensor(hi.numpy(), requires_grad=False) for hi in h)
        T, B = x.shape
        loss = loss_fn(logits, Tensor(y.numpy().reshape(T * B).astype(np.float32)))
        total_loss += loss.numpy().item() * T * B
        total_tokens += T * B
    model.train()
    ppl = math.exp(total_loss / total_tokens)
    return ppl


def train(args):
    print(f"Training LSTM LM on PTB  (hidden={args.hidden_size}, layers={args.num_layers})")

    corpus = Corpus(args.data)
    vocab_size = len(corpus.dictionary)
    print(f"  Vocab size: {vocab_size}")

    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, args.batch_size)

    model = LanguageModel(
        embedding_size=args.embedding_size,
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        seq_model="lstm",
        dropout=args.dropout,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SoftmaxLoss()

    best_val_ppl = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        h = None
        t0 = time.time()

        for i in range(0, train_data.shape[0] - 1, args.seq_len):
            x, y = get_batch(train_data, i, args.seq_len)
            T, B = x.shape

            # Detach hidden state every sequence to bound BPTT
            if h is not None and isinstance(h, tuple):
                h = tuple(Tensor(hi.numpy(), requires_grad=False) for hi in h)

            logits, h = model(x, h)
            targets = Tensor(y.numpy().reshape(T * B).astype(np.float32))
            loss = loss_fn(logits, targets)

            optimizer.reset_grad()
            loss.backward()
            # Gradient clipping
            _clip_grad_norm(model.parameters(), max_norm=0.25)
            optimizer.step()

            total_loss += loss.numpy().item() * T * B
            total_tokens += T * B

        train_ppl = math.exp(total_loss / total_tokens)
        val_ppl = evaluate_ppl(model, val_data, args.seq_len)
        best_val_ppl = min(best_val_ppl, val_ppl)
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"Train PPL: {train_ppl:.1f} | "
            f"Val PPL: {val_ppl:.1f} | "
            f"Best: {best_val_ppl:.1f} | "
            f"Time: {elapsed:.1f}s"
        )

    # Final test set evaluation
    test_data = batchify(corpus.test, args.batch_size)
    test_ppl = evaluate_ppl(model, test_data, args.seq_len)
    print(f"\nTest perplexity: {test_ppl:.1f}")


def _clip_grad_norm(params, max_norm):
    """Clip gradient norms in-place."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            total_norm += (p.grad.numpy() ** 2).sum()
    total_norm = total_norm ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.data = Tensor(p.grad.numpy() * scale, requires_grad=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ptb")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=35)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()
    train(args)