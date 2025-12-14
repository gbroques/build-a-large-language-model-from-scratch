from torch import nn
import torch
from print_utils import print_section


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        tokens = ["Your", "journey", "starts", "with", "one", "step"]
        print_section(
            "2.1 Query Weight Matrix",
            self.W_query.weight,
            [f"dim{i}" for i in range(d_out)],
        )
        print_section(
            "2.2 Key Weight Matrix",
            self.W_key.weight,
            [f"dim{i}" for i in range(d_out)],
        )
        print_section(
            "2.3 Value Weight Matrix",
            self.W_value.weight,
            [f"dim{i}" for i in range(d_out)],
        )
        print_section(
            "2.4 Output Projection Matrix",
            self.out_proj.weight,
            [f"dim{i}" for i in range(d_out)],
        )
        print_section("2.5 Causal Mask", self.mask, tokens, tokens)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        tokens = ["Your", "journey", "starts", "with", "one", "step"]

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        print_section("3.1 Queries (Batch 0)", queries[0], tokens)
        print_section("3.2 Keys (Batch 0)", keys[0], tokens)
        print_section("3.3 Values (Batch 0)", values[0], tokens)

        # Reshape for multi-head attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        print_section(
            "4.1 Reshaped Queries (Batch 0, Head 0)",
            queries[0, :, 0, :],
            tokens,
            calculation=f"Shape: {tuple(queries.shape)} = (batch, tokens, heads, head_dim)",
        )
        print_section("4.2 Reshaped Keys (Batch 0, Head 0)", keys[0, :, 0, :], tokens)
        print_section(
            "4.3 Reshaped Values (Batch 0, Head 0)", values[0, :, 0, :], tokens
        )

        # Transpose for attention computation
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        print_section(
            "5. Attention Scores (Batch 0, Head 0)",
            attn_scores[0, 0],
            tokens,
            tokens,
            calculation=f"Q @ Kt per head: {tuple(queries.shape)} @ transposed",
        )

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        print_section(
            "6. Masked Attention Scores (Batch 0, Head 0)",
            attn_scores[0, 0],
            tokens,
            tokens,
            calculation="Apply causal mask per head",
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        print_section(
            "7. Attention Weights (Batch 0, Head 0)",
            attn_weights[0, 0],
            tokens,
            tokens,
            calculation=f"softmax(Masked Scores / sqrt({keys.shape[-1]}))",
        )

        context_vec = (attn_weights @ values).transpose(1, 2)
        print_section(
            "8. Context Vectors per Head (Batch 0, Head 0)",
            context_vec[0, :, 0, :],
            tokens,
            calculation="Attention @ Values per head, then transpose back",
        )

        # Concatenate heads
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        print_section(
            "9. Concatenated Heads (Batch 0)",
            context_vec[0],
            tokens,
            calculation=f"Reshape to concatenate heads: {tuple(context_vec.shape)}",
        )

        # Final projection
        context_vec = self.out_proj(context_vec)
        print_section(
            "10. Final Context Vectors (Batch 0)",
            context_vec[0],
            tokens,
            calculation="Apply output projection",
        )

        return context_vec


# 6 tokens, with an embedding dimension of 3.
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],
    ]  # step     (x^6)
)

tokens = ["Your", "journey", "starts", "with", "one", "step"]

batch = torch.stack((inputs, inputs), dim=0)
print_section("1. Input Embeddings (Batch 0)", inputs, tokens)

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

batch_labels = [f"B0-{token}" for token in tokens] + [f"B1-{token}" for token in tokens]
print_section(
    "11. Final Output (All Batches)",
    context_vecs.view(-1, d_out),
    batch_labels,
    [f"dim{i}" for i in range(d_out)],
    calculation=f"Output shape: {context_vecs.shape} -> reshaped for display",
)
