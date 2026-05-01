"""
apps/train_cifar.py
====================
Train ResNet-9 on CIFAR-10 using the MiniGrad framework.

Usage:
  python apps/train_cifar.py --data data/cifar-10-batches-py \\
                              --epochs 20 --lr 0.01 --batch_size 128

Expected accuracy after 20 epochs: ~88-91% (varies by seed / hardware).

No PyTorch / TensorFlow — pure MiniGrad.
"""
import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import minigrad
import minigrad.nn as nn
import minigrad.optim as optim
from minigrad.autograd import Tensor
from minigrad.data import CIFAR10Dataset, DataLoader, RandomFlipHorizontal, RandomCrop
from apps.models import ResNet9


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        logits, _ = model(imgs), None  # ResNet9 returns logits directly
        logits = model(imgs)
        preds = np.argmax(logits.numpy(), axis=1)
        lbls = labels.numpy()
        correct += (preds == lbls).sum()
        total += len(lbls)
    model.train()
    return correct / total


def train(args):
    print(f"Training ResNet-9 on CIFAR-10  ({args.epochs} epochs, lr={args.lr})")

    # Data
    transform = lambda x: RandomCrop(4)(RandomFlipHorizontal()(x))
    train_ds = CIFAR10Dataset(args.data, train=True, transform=transform)
    test_ds = CIFAR10Dataset(args.data, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    print(f"  Train size: {len(train_ds)} | Test size: {len(test_ds)}")

    # Model
    model = ResNet9()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loss_fn = nn.SoftmaxLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            optimizer.reset_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.numpy().item()

        test_acc = evaluate(model, test_loader)
        best_acc = max(best_acc, test_acc)
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {epoch_loss / len(train_loader):.4f} | "
            f"Test acc: {100 * test_acc:.2f}% | "
            f"Best: {100 * best_acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

    print(f"\nFinal best test accuracy: {100 * best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cifar-10-batches-py")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    train(args)