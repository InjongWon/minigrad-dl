"""
apps/models.py
==============
Model definitions for showcase applications.

Models:
  - ResNet9: compact 9-layer ResNet for CIFAR-10
  - LanguageModel: LSTM or Transformer language model for PTB

All layers are implemented in MiniGrad — no PyTorch/TensorFlow.
"""
import minigrad.nn as nn
from minigrad.autograd import Tensor
from minigrad.ops.ops_mathematic import reshape, broadcast_to


# ---------------------------------------------------------------------------
# ResNet-9 for CIFAR-10
# ---------------------------------------------------------------------------
def conv_bn_relu(in_c, out_c, kernel=3, stride=1):
    """Conv + BN + ReLU block."""
    return nn.Sequential(
        nn.Conv(in_c, out_c, kernel, stride=stride),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


class ResidualBlock(nn.Module):
    """Two conv_bn_relu blocks with a skip connection."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu(channels, channels),
            conv_bn_relu(channels, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ResNet9(nn.Module):
    """
    9-layer ResNet for CIFAR-10 (10 classes, 32×32 images).

    Architecture:
      conv_bn_relu(3  → 16)
      conv_bn_relu(16 → 32, stride=2)
      ResidualBlock(32)
      conv_bn_relu(32 → 64, stride=2)
      conv_bn_relu(64 → 128, stride=2)
      ResidualBlock(128)
      GlobalAvgPool
      Linear(128 → 10)
    """

    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.block1 = conv_bn_relu(3, 16)
        self.block2 = conv_bn_relu(16, 32, stride=2)
        self.res1 = ResidualBlock(32)
        self.block3 = conv_bn_relu(32, 64, stride=2)
        self.block4 = conv_bn_relu(64, 128, stride=2)
        self.res2 = ResidualBlock(128)
        self.classifier = nn.Linear(128, 10, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, 32, 32)
        x = self.block1(x)
        x = self.block2(x)
        x = self.res1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.res2(x)
        # Global average pool: (N, C, H, W) -> (N, C)
        N, C, H, W = x.shape
        x = x.sum(axes=(2, 3)) / (H * W)
        x = reshape(x, (N, C))
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Language Model (LSTM or Transformer)
# ---------------------------------------------------------------------------
class LanguageModel(nn.Module):
    """
    Word-level language model for Penn Treebank.

    Architecture:
      Embedding(vocab_size, embedding_size)
      [LSTM | Transformer](embedding_size, hidden_size, num_layers)
      Linear(hidden_size, vocab_size)  [tied weights optional]

    Usage:
      model = LanguageModel(embedding_size=128, vocab_size=10000,
                            hidden_size=256, num_layers=2,
                            seq_model='lstm')
      logits, h = model(x, h)  # x: (seq_len, batch)
    """

    def __init__(
        self,
        embedding_size: int,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 1,
        seq_model: str = "lstm",   # 'lstm' or 'transformer'
        seq_len: int = 35,
        dropout: float = 0.0,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_model = seq_model

        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      device=device, dtype=dtype)

        if seq_model == "lstm":
            self.model = nn.LSTM(embedding_size, hidden_size,
                                 num_layers=num_layers, device=device, dtype=dtype)
            head_dim = hidden_size
        elif seq_model == "transformer":
            # Transformer head_dim = embedding_size (positional embeddings on top)
            num_heads = max(1, embedding_size // 64)
            self.model = nn.Transformer(
                q_features=embedding_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=hidden_size,
                seq_len=seq_len,
                dropout=dropout,
                causal=True,
                device=device,
                dtype=dtype,
            )
            head_dim = embedding_size
        else:
            raise ValueError(f"Unknown seq_model: {seq_model}")

        self.linear = nn.Linear(head_dim, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Tensor, h=None):
        """
        x: (seq_len, batch) integer token ids
        h: initial hidden state (LSTM only)

        Returns:
          logits: (seq_len * batch, vocab_size)
          h_out:  final hidden state (LSTM) or None (Transformer)
        """
        T, B = x.shape
        emb = self.embedding(x)  # (T, B, E)

        if self.seq_model == "lstm":
            out, h_out = self.model(emb, h)  # (T, B, H)
            logits = self.linear(reshape(out, (T * B, self.hidden_size)))
        else:
            # Transformer expects (B, T, E)
            emb_t = emb.transpose((1, 0, 2))  # (B, T, E)
            out = self.model(emb_t)            # (B, T, E)
            out_t = out.transpose((1, 0, 2))   # (T, B, E)
            logits = self.linear(reshape(out_t, (T * B, self.embedding_size)))
            h_out = None

        return logits, h_out