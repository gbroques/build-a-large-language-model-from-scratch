from torch import nn
import torch
from print_utils import print_section


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # automatically move mask to the appropriate device (CPU or GPU) along with our model
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        tokens = ["Your", "journey", "starts", "with", "one", "step"]
        print_section("2.1 Query Weight Matrix", self.W_query.weight, ["dim0", "dim1"])
        print_section("2.2 Key Weight Matrix", self.W_key.weight, ["dim0", "dim1"])
        print_section("2.3 Value Weight Matrix", self.W_value.weight, ["dim0", "dim1"])
        print_section("2.4 Causal Mask", self.mask, tokens, tokens)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        tokens = ["Your", "journey", "starts", "with", "one", "step"]

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        print_section("3.1 Queries (Batch 0)", queries[0], tokens)
        print_section("3.2 Keys (Batch 0)", keys[0], tokens)
        print_section("3.3 Values (Batch 0)", values[0], tokens)

        attn_scores = queries @ keys.transpose(1, 2)
        print_section(
            "4. Attention Scores (Batch 0)",
            attn_scores[0],
            tokens,
            tokens,
            calculation=f"Q @ Kt = {tuple(queries.shape)} @ {tuple(keys.transpose(1, 2).shape)}",
        )

        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        print_section(
            "5. Masked Attention Scores (Batch 0)",
            attn_scores[0],
            tokens,
            tokens,
            calculation="Apply causal mask (set future positions to -inf)",
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        print_section(
            "6. Attention Weights (Batch 0)",
            attn_weights[0],
            tokens,
            tokens,
            calculation=f"softmax(Masked Scores / sqrt({keys.shape[-1]}))",
        )

        context_vec = attn_weights @ values
        print_section(
            "7. Context Vectors (Batch 0)",
            context_vec[0],
            tokens,
            ["dim0", "dim1"],
            calculation=f"Attention Weights @ Values = {tuple(attn_weights.shape)} @ {tuple(values.shape)}",
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
context_length = batch.shape[1]
d_in = inputs.shape[1]
d_out = 2
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
batch_labels = [f"B0-{token}" for token in tokens] + [f"B1-{token}" for token in tokens]
print_section(
    "8. Final Context Vectors (All Batches)",
    context_vecs.view(-1, d_out),
    batch_labels,
    ["dim0", "dim1"],
    calculation=f"Output shape: {context_vecs.shape} -> reshaped for display",
)
