import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        assert output_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * output_dim)
        self.o_proj = nn.Linear(output_dim, output_dim)

        self._reset_parameters()

    def _reset_parameters(self):

        with torch.no_grad():
            nn.init.xavier_uniform_(self.qkv_proj.weight)
            self.qkv_proj.bias.data.uniform_(0., 0.1)

            nn.init.xavier_uniform_(self.o_proj.weight)
            self.o_proj.bias.data.uniform_(0., 0.1)

    def _scaled_attention(self, q, k, v, mask=None):

        d_k = q.size()[-1]

        sqrt_d_k = math.sqrt(d_k)

        att_logits = torch.matmul(q, k.transpose(-2, -1))
        att_logits /= sqrt_d_k

        if mask is not None:
            att_logits = att_logits.masked_fill(mask == 0, -9e15)

        attn_scores = F.softmax(att_logits, dim=-1)
        values = torch.matmul(attn_scores, v)

        return values, attn_scores

    def forward(self, x, mask=None, return_attention=False):

        batch_size, seq_length, embed_dim = x.size()

        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self._scaled_attention(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.output_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention

        return o


class MHEncoder(nn.Module):

    def __init__(self, input_dim, num_heads, ln_dim, dropout=0.0):
        super().__init__()

        self._multi_attn = MultiHeadAttention(input_dim, input_dim, num_heads)

        self._linear_net = nn.Sequential(
            nn.Linear(input_dim, ln_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ln_dim, input_dim)
        )

        # norm layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + attn_out
        x = self.norm1(x)
        x = self.dropout(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + linear_out
        x = self.norm2(x)
        x = self.dropout(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, emb_dim, max_len=5000):
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
