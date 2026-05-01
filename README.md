# MiniGrad — Deep Learning System from Scratch

> **MiniGrad** is a PyTorch-like deep learning system built from scratch — including a C++/CUDA n-dimensional array backend, reverse-mode automatic differentiation, a full neural network library, and trained models achieving competitive accuracy on CIFAR-10 and Penn Treebank. No PyTorch. No TensorFlow. Just NumPy for data loading.

Built as part of CMU 10-714: Deep Learning Systems.

---

## Architecture

```
minigrad/
├── minigrad/
│   ├── autograd.py          # Tensor, Op, computational graph (HW1)
│   ├── ops/                 # All forward + backward ops (HW1 + HW4)
│   │   ├── ops_mathematic.py
│   │   └── ops_logarithmic.py
│   ├── backend_ndarray/     # Custom NDArray (HW3)
│   │   ├── ndarray.py           # Python strides/reshape/broadcast
│   │   ├── ndarray_backend_cpu.cc   # C++ compact, matmul, ewise
│   │   └── ndarray_backend_cuda.cu  # CUDA kernels
│   ├── nn/
│   │   ├── nn_basic.py      # Linear, BN, LayerNorm, Dropout (HW2)
│   │   ├── nn_conv.py       # Conv2d (HW4)
│   │   ├── nn_sequence.py   # RNN, LSTM (HW4)
│   │   └── nn_transformer.py # MultiHeadAttention, Transformer (HW4 extra)
│   ├── init/                # Xavier, Kaiming initializers (HW2)
│   ├── optim/               # SGD, Adam (HW2)
│   └── data/                # Dataset, DataLoader, CIFAR-10, PTB (HW4)
├── apps/
│   ├── train_cifar.py       # ResNet-9 → CIFAR-10
│   ├── train_lm_lstm.py     # LSTM language model on PTB
│   ├── train_lm_transformer.py  # Transformer LM on PTB
│   └── models.py            # ResNet9, LanguageModel definitions
├── benchmarks/
│   └── matmul_benchmark.py  # CPU/CUDA matmul speed vs numpy
├── tests/
│   ├── test_autograd.py
│   ├── test_ndarray.py
│   ├── test_nn.py
│   └── test_ops.py
└── notebooks/
    └── autograd_explained.ipynb
```

---

## Module Breakdown & Skills Demonstrated

### 1 · Autograd Engine (`minigrad/autograd.py`)
Implements the core reverse-mode automatic differentiation framework.

**Key concepts:**
- `Tensor` class wrapping raw data with a computation graph
- `Op` base class with `compute()` (forward) and `gradient()` (backward)
- Topological sort of the compute graph for correct gradient accumulation
- `compute_gradient_of_variables()` — the reverse AD algorithm
- Higher-order gradients supported (gradient graph is itself a Tensor graph)

### 2 · Ops (`minigrad/ops/`)
All math operators with analytical backward passes:

| Op | Forward | Backward |
|----|---------|----------|
| `EWiseAdd`, `AddScalar` | element-wise add | pass-through |
| `EWiseMul`, `MulScalar` | element-wise multiply | chain rule × other input |
| `MatMul` | matrix product | `out_grad @ B.T`, `A.T @ out_grad` |
| `Summation` | sum over axes | broadcast grad back |
| `BroadcastTo` | broadcast | sum over broadcast axes |
| `Reshape`, `Transpose` | shape/axis manipulation | inverse reshape/transpose |
| `Log`, `Exp` | log, exp | `1/x * grad`, `exp(x) * grad` |
| `ReLU` | max(0, x) | indicator function |
| `LogSumExp` | numerically-stable log-sum-exp | softmax gradient |
| `Tanh` | tanh | `(1 - tanh²(x)) * grad` |
| `Stack` / `Split` | concat on new axis | inverse of each other |
| `Flip`, `Dilate`, `Conv` | image ops | via transposed conv |

### 3 · NDArray Backend (`minigrad/backend_ndarray/`)
A custom n-dimensional array library with CPU and CUDA backends.

**Python layer** handles all structural operations without copying:
- `reshape()`, `permute()`, `broadcast_to()`, `__getitem__()` — stride manipulation only
- `compact()` triggers the C++ copy into contiguous memory

**C++ backend** (`ndarray_backend_cpu.cc`):
- `Compact()` / `EwiseSetitem()` / `ScalarSetitem()` — stride-aware memory copy
- Element-wise ops, reductions
- `Matmul()` with cache-friendly tiled implementation (`TILE × TILE` blocks)

**CUDA backend** (`ndarray_backend_cuda.cu`):
- All of the above ported to CUDA kernels
- Parallelized over output elements

### 4 · Neural Network Library (`minigrad/nn/`)

**`nn_basic.py`** (HW2):
- `Linear` — `y = xW^T + b`, Kaiming Uniform init
- `ReLU` — wraps the op
- `Sequential` — composes modules
- `LayerNorm1d` — `(x - mean) / std * w + b`
- `BatchNorm1d` — running stats, train/eval modes
- `Dropout` — Bernoulli mask
- `Softmax`, `SoftmaxLoss` — uses numerically stable LogSumExp

**`nn_conv.py`** (HW4):
- `Conv` — 2D convolution using im2col + matmul
- `BatchNorm2d` — wraps `BatchNorm1d` for spatial dims

**`nn_sequence.py`** (HW4):
- `RNN` / `RNNCell` — vanilla recurrent network
- `LSTM` / `LSTMCell` — long short-term memory with forget/input/output/cell gates
- `Embedding` — lookup table

**`nn_transformer.py`** (HW4 extra):
- `MultiHeadAttention` — scaled dot-product attention with causal masking
- `AttentionLayer` — prenorm multi-head self/cross attention
- `TransformerLayer` — attention + 2-layer MLP residual block
- `Transformer` — stacked layers + positional embeddings

### 5 · Initializers (`minigrad/init/`)
- `xavier_uniform`, `xavier_normal` — [Glorot et al., 2010](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
- `kaiming_uniform`, `kaiming_normal` — [He et al., 2015](https://arxiv.org/pdf/1502.01852.pdf)

### 6 · Optimizers (`minigrad/optim/`)
- `SGD` with optional momentum and weight decay
- `Adam` — adaptive moment estimation

### 7 · Data Pipeline (`minigrad/data/`)
- `Dataset` / `DataLoader` — iterable batching with shuffle
- `CIFAR10Dataset` — reads raw binary files, returns `(3, 32, 32)` tensors
- `Corpus` / PTB tokenizer — character-level Penn Treebank loader
- `RandomCrop`, `RandomFlipHorizontal` — data augmentation transforms

---

## Showcase Applications

### ResNet-9 on CIFAR-10 (`apps/train_cifar.py`)

```
ResNet9
  Block 1: Conv(3→16, 3×3) + BN + ReLU
  Block 2: Conv(16→32, 3×3, stride=2) + BN + ReLU
  Residual Block 1: Conv(32→32, 3×3) + BN + ReLU, Conv(32→32, 3×3) + BN + ReLU
  Block 3: Conv(32→64, 3×3, stride=2) + BN + ReLU
  Block 4: Conv(64→128, 3×3, stride=2) + BN + ReLU
  Residual Block 2: ...
  FC: 128 → 10
```

Run training:
```bash
python apps/train_cifar.py --epochs 20 --lr 0.01 --batch_size 128 --device cuda
```

### LSTM Language Model on PTB (`apps/train_lm_lstm.py`)

```bash
python apps/train_lm_lstm.py --epochs 20 --lr 0.003 --seq_len 35 --hidden_size 256
```

### Transformer Language Model on PTB (`apps/train_lm_transformer.py`)

```bash
python apps/train_lm_transformer.py --epochs 20 --lr 0.003 --seq_len 20 --num_layers 4
```

---

## Matmul Benchmark (`benchmarks/matmul_benchmark.py`)

Compares our tiled C++ matmul against NumPy and (optionally) PyTorch:

```bash
python benchmarks/matmul_benchmark.py
```

Sample output (approximate):
```
Matrix size: 1024 × 1024
  NumPy (MKL):     12.3 ms
  MiniGrad CPU:    28.1 ms   (2.3× slower — no BLAS, pure C++)
  MiniGrad CUDA:    4.1 ms   (3.0× faster than NumPy)
```

---

## Setup

```bash
# Build C++ backends
make

# Install dependencies
pip install numpy pytest matplotlib

# Run all tests
pytest tests/ -v
```

---

## Key Takeaways

| Skill | Where demonstrated |
|-------|-------------------|
| Reverse-mode autodiff from scratch | `minigrad/autograd.py` |
| Analytical gradients for 15+ ops | `minigrad/ops/` |
| C++ memory management & strides | `backend_ndarray/ndarray_backend_cpu.cc` |
| CUDA kernel programming | `backend_ndarray/ndarray_backend_cuda.cu` |
| Neural network module system | `minigrad/nn/` |
| Conv, BN, dropout, layer norm | `minigrad/nn/nn_basic.py`, `nn_conv.py` |
| RNN, LSTM, Transformer | `minigrad/nn/nn_sequence.py`, `nn_transformer.py` |
| SGD, Adam optimizers | `minigrad/optim/` |
| Data loading pipeline | `minigrad/data/` |
| End-to-end model training | `apps/` |

---

## References

- [Deep Learning Systems course (CMU 10-714)](https://dlsyscourse.org/)
- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [Understanding the difficulty of training deep feedforward neural networks — Glorot & Bengio, 2010](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
- [Delving Deep into Rectifiers — He et al., 2015](https://arxiv.org/pdf/1502.01852.pdf)
- [Layer Normalization — Ba et al., 2016](https://arxiv.org/abs/1607.06450)