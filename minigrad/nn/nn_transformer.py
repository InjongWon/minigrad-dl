"""
minigrad/nn/nn_transformer.py
==============================
Transformer architecture (decoder-only, prenorm variant).

Implements:
  1. MultiHeadAttention — scaled dot-product attention with causal masking
  2. AttentionLayer — prenorm self/cross attention with projections
  3. TransformerLayer — attention + 2-layer MLP residual block
  4. Transformer — stacked layers + learnable positional embeddings

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           https://arxiv.org/abs/1706.03762

Prenorm variant: "On Layer Normalization in the Transformer Architecture"
           https://arxiv.org/abs/2002.04745

Based on CMU 10-714 HW4 extra.
"""
import numpy as np
from minigrad.autograd import Tensor
import minigrad.init as init
from minigrad.nn.nn_basic import (
    Module, Parameter, Linear, Dropout, LayerNorm1d, Embedding, Sequential, ReLU
)
from minigrad.ops.ops_mathematic import (
    matmul, reshape, broadcast_to, summation, stack
)
from minigrad.ops.ops_logarithmic import logsumexp


# ---------------------------------------------------------------------------
# Softmax helper (not a full TensorOp — just for inference)
# ---------------------------------------------------------------------------
def _softmax(x: Tensor) -> Tensor:
    """Row-wise softmax."""
    from minigrad.ops.ops_mathematic import exp
    M = Tensor(np.max(x.numpy(), axis=-1, keepdims=True), requires_grad=False)
    e = exp(x - broadcast_to(M, x.shape))
    s = broadcast_to(reshape(summation(e, axes=(-1,)), e.shape[:-1] + (1,)), e.shape)
    return e / s


# ---------------------------------------------------------------------------
# 1. MultiHeadAttention (activation layer, no trainable params)
# ---------------------------------------------------------------------------
class MultiHeadAttention(Module):
    """
    Scaled dot-product multi-head attention activation.

    Given Q, K, V of shape (B, H, T, D):
      X = softmax(QK^T / sqrt(D)) V

    Supports:
      - Causal masking (decoder / auto-regressive)
      - Dropout on attention weights
    """

    def __init__(self, head_dim, num_heads, dropout=0.0, causal=False,
                 device=None, dtype="float32"):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        self.causal = causal

    def create_causal_mask(self, T: int) -> np.ndarray:
        """Upper-triangular boolean mask (True = masked)."""
        return np.triu(np.ones((T, T), dtype=bool), k=1)

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Batched matrix multiply for (B, H, T, D) tensors."""
        B, H, T, D = a.shape
        a_2d = reshape(a, (B * H, T, D))
        b_2d = reshape(b, (B * H, D, T))
        out = matmul(reshape(a_2d, (B * H * T, D)),
                     reshape(b_2d, (D, B * H * T)))
        # Actually we need (B*H, T, T) — do it properly
        out_list = []
        for i in range(B * H):
            ai = a_2d[i]           # (T, D)
            bi = b_2d[i]           # (D, T)
            out_list.append(matmul(ai, bi))   # (T, T)
        return stack(out_list, axis=0).reshape((B, H, T, T))

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """
        Q, K, V: (B, H, T, D)
        Returns: (B, H, T, D)
        """
        D = self.head_dim
        T = Q.shape[2]
        scale = D ** 0.5

        # Attention scores: (B, H, T, T)
        scores = self.matmul(Q, K.transpose((0, 1, 3, 2))) / scale

        if self.causal:
            mask = self.create_causal_mask(T)
            # Set masked positions to -1e9
            mask_val = -1e9 * mask.astype(np.float32)
            mask_t = broadcast_to(
                Tensor(mask_val[np.newaxis, np.newaxis], requires_grad=False),
                scores.shape
            )
            scores = scores + mask_t

        probs = _softmax(scores)  # (B, H, T, T)
        probs = self.dropout(probs)

        # Weighted sum of values
        out = self.matmul(probs, V)   # (B, H, T, D)
        self.probs = probs
        return out


# ---------------------------------------------------------------------------
# 2. AttentionLayer (with learned projections + prenorm)
# ---------------------------------------------------------------------------
class AttentionLayer(Module):
    """
    Multi-head self (or cross) attention layer with prenorm.

    Self-attention when k=None, v=None.
    Cross-attention when k and v are provided.

    Step-by-step:
      1. Q' = LayerNorm_q(Q) W_q
         K' = LayerNorm_k(K) W_k
         V' = LayerNorm_v(V) W_v
      2. Unravel H heads from channel dim
      3. X = MultiHeadAttention(Q', K', V')
      4. Rearrange H heads back, project: X' = X W_o
    """

    def __init__(self, q_features, num_heads, k_features=None, v_features=None,
                 dropout=0.0, causal=False, device=None, dtype="float32"):
        super().__init__()
        self.q_features = q_features
        self.num_heads = num_heads
        self.head_dim = q_features // num_heads

        k_features = k_features or q_features
        v_features = v_features or q_features

        # Prenorm
        self.prenorm_q = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(v_features, device=device, dtype=dtype)

        # Projection matrices
        self.W_q = Linear(q_features, q_features, bias=False, device=device, dtype=dtype)
        self.W_k = Linear(k_features, q_features, bias=False, device=device, dtype=dtype)
        self.W_v = Linear(v_features, q_features, bias=False, device=device, dtype=dtype)
        self.out_projection = Linear(q_features, q_features, bias=False, device=device, dtype=dtype)

        self.attn = MultiHeadAttention(self.head_dim, num_heads,
                                       dropout=dropout, causal=causal)

    def forward(self, q: Tensor, k: Tensor = None, v: Tensor = None) -> Tensor:
        """
        q: (B, T, D_q)
        k, v: (B, T, D_k/D_v) or None (self-attention)
        """
        if k is None:
            k = q
        if v is None:
            v = q

        B, T, _ = q.shape
        H = self.num_heads
        D = self.head_dim

        # Apply prenorm and project
        def project(x, norm, proj):
            # x: (B, T, D)  ->  norm over last dim  -> proj
            x_2d = reshape(x, (B * T, x.shape[-1]))
            x_n = norm(x_2d)
            x_p = proj(x_n)
            return reshape(x_p, (B, T, H * D))

        Q_p = project(q, self.prenorm_q, self.W_q)
        K_p = project(k, self.prenorm_k, self.W_k)
        V_p = project(v, self.prenorm_v, self.W_v)

        # Unravel heads: (B, T, H*D) -> (B, H, T, D)
        def unravel(x):
            return reshape(x, (B, T, H, D)).transpose((0, 2, 1, 3))

        Q_h = unravel(Q_p)
        K_h = unravel(K_p)
        V_h = unravel(V_p)

        # Multi-head attention
        X = self.attn(Q_h, K_h, V_h)  # (B, H, T, D)

        # Rearrange back: (B, H, T, D) -> (B, T, H*D)
        X = X.transpose((0, 2, 1, 3))      # (B, T, H, D)
        X = reshape(X, (B, T, H * D))

        # Output projection
        X_2d = reshape(X, (B * T, H * D))
        out = self.out_projection(X_2d)
        return reshape(out, (B, T, H * D))


# ---------------------------------------------------------------------------
# 3. TransformerLayer (prenorm residual block)
# ---------------------------------------------------------------------------
class TransformerLayer(Module):
    """
    Prenorm Transformer residual block:

      x = x + Dropout(Attention(x))
      x = x + Dropout(Linear2(Dropout(ReLU(Linear1(LayerNorm(x))))))
    """

    def __init__(self, q_features, num_heads, dim_feedforward,
                 dropout=0.0, causal=False, device=None, dtype="float32"):
        super().__init__()
        self.attention = AttentionLayer(
            q_features, num_heads, dropout=dropout, causal=causal,
            device=device, dtype=dtype
        )
        self.norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.ff = Sequential(
            Linear(q_features, dim_feedforward, device=device, dtype=dtype),
            ReLU(),
            Dropout(dropout),
            Linear(dim_feedforward, q_features, device=device, dtype=dtype),
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        x = x + self.dropout1(self.attention(x))
        # Feedforward with prenorm
        x_2d = reshape(x, (B * T, D))
        x_normed = self.norm(x_2d)
        ff_out = self.ff(x_normed)
        x = x + self.dropout2(reshape(ff_out, (B, T, D)))
        return x


# ---------------------------------------------------------------------------
# 4. Transformer model
# ---------------------------------------------------------------------------
class Transformer(Module):
    """
    Decoder-only Transformer with learned positional embeddings.

    Input: integer token indices (B, T)
    Output: hidden states (B, T, D)
    """

    def __init__(self, q_features, num_heads, num_layers, dim_feedforward,
                 seq_len, dropout=0.0, causal=True, device=None, dtype="float32"):
        super().__init__()
        self.layers = [
            TransformerLayer(
                q_features, num_heads, dim_feedforward,
                dropout=dropout, causal=causal,
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ]
        self.pos_embedding = Embedding(seq_len, q_features, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T) integer token ids
        Returns: (B, T, D)
        """
        B, T = x.shape
        positions = Tensor(
            np.arange(T)[np.newaxis].repeat(B, axis=0), requires_grad=False
        )
        # Token embeddings must be provided externally (by LanguageModel)
        # Here we just add positional embeddings to whatever x already is
        # (LanguageModel passes in the token embedding tensor)
        pos_emb = self.pos_embedding(positions)  # (B, T, D)
        x = x + pos_emb

        for layer in self.layers:
            x = layer(x)
        return x