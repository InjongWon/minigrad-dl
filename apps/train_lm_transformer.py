"""
apps/train_lm_transformer.py
=============================
Train a Transformer language model on Penn Treebank.

Usage:
  python apps/train_lm_transformer.py --data data/ptb \\
                                        --epochs 20 --lr 0.003 \\
                                        --num_layers 4 --seq_len 20

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
    for i in range(0, data.shape[0] - 1, seq_len):
        x, y = get_batch(data, i, seq_len)
        T, B = x.shape
        logits, _ = model(x)
        targets = Tensor(y.numpy().reshape(T * B).astype(np.float32))
        loss = loss_fn(logits, targets)
        total_loss += loss.numpy().item() * T * B
        total_tokens += T * B
    model.train()
    return math.exp(total_loss / total_tokens)


def train(args):
    print(f"Training Transformer LM on PTB  (layers={args.num_layers})")

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
        seq_model="transformer",
        seq_len=args.seq_len,
        dropout=args.dropout,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SoftmaxLoss()

    best_val_ppl = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()

        for i in range(0, train_data.shape[0] - 1, args.seq_len):
            x, y = get_batch(train_data, i, args.seq_len)
            T, B = x.shape

            logits, _ = model(x)
            targets = Tensor(y.numpy().reshape(T * B).astype(np.float32))
            loss = loss_fn(logits, targets)

            optimizer.reset_grad()
            loss.backward()
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

    test_data = batchify(corpus.test, args.batch_size)
    test_ppl = evaluate_ppl(model, test_data, args.seq_len)
    print(f"\nTest perplexity: {test_ppl:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ptb")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    train(args)