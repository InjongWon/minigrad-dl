"""
minigrad/data/__init__.py
=========================
Dataset, DataLoader, CIFAR-10, Penn Treebank, augmentation transforms.

Based on CMU 10-714 HW4.
"""
import numpy as np
import os
import pickle
import struct
import gzip
from minigrad.autograd import Tensor


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------
class Dataset:
    """Abstract base class for datasets."""

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DataLoader:
    """
    Iterates over a Dataset in mini-batches, with optional shuffling.

    Returns Tensors for each batch element.
    """

    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, self.batch_size):
            batch_idx = indices[start: start + self.batch_size]
            samples = [self.dataset[i] for i in batch_idx]
            # Each sample can be a tuple or a single array
            if isinstance(samples[0], (tuple, list)):
                yield tuple(
                    Tensor(np.stack([s[j] for s in samples]))
                    for j in range(len(samples[0]))
                )
            else:
                yield Tensor(np.stack(samples))

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class RandomFlipHorizontal:
    """Randomly flip a (C, H, W) image horizontally with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return img[:, :, ::-1].copy()
        return img


class RandomCrop:
    """
    Randomly crop a (C, H, W) image by padding then cropping.
    Equivalent to torchvision RandomCrop with padding.
    """

    def __init__(self, padding: int):
        self.padding = padding

    def __call__(self, img: np.ndarray) -> np.ndarray:
        C, H, W = img.shape
        p = self.padding
        padded = np.pad(img, ((0, 0), (p, p), (p, p)), mode="constant")
        top = np.random.randint(0, 2 * p + 1)
        left = np.random.randint(0, 2 * p + 1)
        return padded[:, top: top + H, left: left + W].copy()


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------
class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset.

    Downloads from:
      https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

    Returns: (image, label) where image is (3, 32, 32) float32 [0,1].
    """

    def __init__(self, base_folder: str, train: bool = True, transform=None):
        self.transform = transform
        if train:
            filenames = [
                os.path.join(base_folder, f"data_batch_{i}") for i in range(1, 6)
            ]
        else:
            filenames = [os.path.join(base_folder, "test_batch")]

        images, labels = [], []
        for fn in filenames:
            with open(fn, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            imgs = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            lbls = np.array(batch[b"labels"], dtype=np.int64)
            images.append(imgs)
            labels.append(lbls)

        self.images = np.concatenate(images)
        self.labels = np.concatenate(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# Penn Treebank corpus
# ---------------------------------------------------------------------------
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    """Penn Treebank tokenizer."""

    def __init__(self, path: str):
        self.dictionary = Dictionary()
        self.train = self._tokenize(os.path.join(path, "train.txt"))
        self.valid = self._tokenize(os.path.join(path, "valid.txt"))
        self.test = self._tokenize(os.path.join(path, "test.txt"))

    def _tokenize(self, path: str) -> np.ndarray:
        assert os.path.exists(path), f"File not found: {path}"
        with open(path, encoding="utf8") as f:
            ids = []
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)
                    ids.append(self.dictionary.word2idx[word])
        return np.array(ids, dtype=np.int64)


def batchify(data: np.ndarray, batch_size: int, device=None, dtype="float32"):
    """
    Reshape flat token array into (num_batches, batch_size) for LM training.
    Truncates remainder.
    """
    n_batch = len(data) // batch_size
    data = data[: n_batch * batch_size]
    data = data.reshape(batch_size, n_batch).T  # (n_batch, batch_size)
    return Tensor(data.astype(dtype), device=device)


def get_batch(data: Tensor, i: int, seq_len: int):
    """
    Get a (seq_len, batch_size) slice starting at column i.
    Returns (inputs, targets).
    """
    n = data.shape[0]
    sl = min(seq_len, n - 1 - i)
    x = data[i: i + sl]
    y = data[i + 1: i + sl + 1]
    return x, y


# ---------------------------------------------------------------------------
# MNIST (for HW0-style tests)
# ---------------------------------------------------------------------------
def parse_mnist(image_filename: str, label_filename: str):
    """
    Parse MNIST binary files.

    Returns:
      X: (N, 784) float32 in [0, 1]
      y: (N,) uint8 labels
    """
    with gzip.open(image_filename, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    with gzip.open(label_filename, "rb") as f:
        f.read(8)
        y = np.frombuffer(f.read(), dtype=np.uint8)
    return X.astype(np.float32) / 255.0, y