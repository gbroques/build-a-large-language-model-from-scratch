import torch
import torch.nn as nn
from print_utils import print_section, print_section_header, print_matrix


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

print_section("1. Token Embeddings", inputs, tokens)


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # nn.Linear has an optimized weight initialization scheme for more stable and effective model training.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        print_section("2.1 Query Weight Matrix", self.W_query.weight, ["dim0", "dim1"])
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        print_section("2.2 Key Weight Matrix", self.W_key.weight, ["dim0", "dim1"])
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        print_section("2.3 Value Weight Matrix", self.W_value.weight, ["dim0", "dim1"])

    def forward(self, x):
        queries = self.W_query(x)
        print_section("3.1 Queries", queries, tokens)
        keys = self.W_key(x)
        print_section("3.2 Keys", keys, tokens)
        values = self.W_value(x)
        print_section("3.3 Values", values, tokens)

        attention_scores = queries @ keys.T  # omega
        print_section(
            "4. Attention Scores", 
            attention_scores, 
            tokens, 
            tokens,
            calculation=f"Q @ Kt = {tuple(queries.shape)} @ {tuple(keys.T.shape)}"
        )

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        print_section(
            "5. Attention Weights", 
            attention_weights, 
            tokens, 
            tokens,
            calculation=f"softmax(Attention Scores / sqrt(dim_k)) = softmax(Attention Scores / sqrt({keys.shape[-1]}))"
        )

        context_vectors = attention_weights @ values
        print_section(
            "6. Context Vectors", 
            context_vectors, 
            tokens, 
            ["dim0", "dim1"],
            calculation=f"Attention Weights @ Values = {tuple(attention_weights.shape)} @ {tuple(values.shape)}"
        )

        return context_vectors


torch.manual_seed(789)
d_in = inputs.shape[1]
d_out = 2
sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v2(inputs)
