import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention (nn.Module):

    def __init__(self, embed_dim):
        super(SingleHeadSelfAttention, self).__init__()

        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def  forward(self, queries, keys, values):
        # 计算 Q, K, V
        q = self.query(queries)
        k = self.key(keys)
        v = self.value(values)

        energy = torch.matmul(q, k.transpose(-1, -2))  # Shape (N, seq_len, seq_len)
        scaling_factor = self.embed_size ** 0.5
        energy = energy / scaling_factor

        attention = F.softmax(energy, dim=-1)

        output = torch.matmul(attention, v)

        return output, attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, queries, keys, values):
        N, seq_len, embed_dim = queries.shape

        # Compute Q, K, V
        q = self.query(queries)  # (N, seq_len, embed_dim)
        k = self.key(keys)      # (N, seq_len, embed_dim)
        v = self.value(values)  # (N, seq_len, embed_dim)

        # Split into multiple heads
        q = q.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, seq_len, head_dim)
        k = k.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, seq_len, head_dim)
        v = v.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, seq_len, head_dim)

        # Compute scaled dot-product attention
        energy = torch.matmul(q, k.transpose(-1, -2))  # (N, num_heads, seq_len, seq_len)
        scaling_factor = self.head_dim ** 0.5
        energy = energy / scaling_factor

        attention = F.softmax(energy, dim=-1)  # (N, num_heads, seq_len, seq_len)

        # Apply attention to values
        out = torch.matmul(attention, v)  # (N, num_heads, seq_len, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(N, seq_len, self.embed_dim)  # (N, seq_len, embed_dim)

        # Final linear layer
        out = self.out(out)  # (N, seq_len, embed_dim)

        return out, attention

