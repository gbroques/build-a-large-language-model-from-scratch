import torch
import torch.nn as nn
from typing import List, Optional




def print_section_header(title: str, shape: torch.Size) -> None:
    """Print formatted section header with title and shape."""
    heading = f"{title} {tuple(shape)}"
    print(heading)
    print("-" * len(heading))


def print_matrix(
    data: torch.Tensor,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    precision: int = 2,
) -> None:
    """Print formatted matrix data with row and optional column labels."""
    label_width = max(len(label) for label in (row_labels or col_labels or [""])) + 1
    if col_labels:
        print(
            " " * label_width,
            " ".join(f"{label:>{label_width}}" for label in col_labels),
        )
    for i in range(data.shape[0]):
        if row_labels:
            print(
                f"{row_labels[i]:>{label_width}}",
                " ".join(
                    f"{data[i][j]:{label_width}.{precision}f}"
                    for j in range(data.shape[1])
                ),
            )
        else:
            print(
                " ".join(
                    f"{data[i][j]:{label_width}.{precision}f}"
                    for j in range(data.shape[1])
                )
            )


def print_section(
    title: str,
    data: torch.Tensor,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    precision: int = 2,
) -> None:
    """Print formatted tensor data with row and optional column labels."""
    print_section_header(title, data.shape)
    print_matrix(data, row_labels, col_labels, precision)
    print()


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

print_section("Token Embeddings", inputs, tokens)


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # nn.Linear has an optimized weight initialization scheme for more stable and effective model training.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        print_section("Query Weight Matrix", self.W_query.weight, ["dim0", "dim1"])
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        print_section("Key Weight Matrix", self.W_key.weight, ["dim0", "dim1"])
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        print_section("Value Weight Matrix", self.W_value.weight, ["dim0", "dim1"])

    def forward(self, x):
        queries = self.W_query(x)
        print_section("Queries", queries, tokens)
        keys = self.W_key(x)
        print_section("Keys", keys, tokens)
        values = self.W_value(x)
        print_section("Values", values, tokens)

        attention_scores = queries @ keys.T  # omega
        print_section_header("Attention Scores", attention_scores.shape)
        print(f"Q @ Kt = {tuple(queries.shape)} @ {(tuple(keys.T.shape))}\n")
        print_matrix(attention_scores, tokens, tokens)
        print()

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        print_section_header("Attention Weights", attention_weights.shape)
        print(
            f"softmax(Attention Scores / sqrt(dim_k)) = softmax(Attention Scores / sqrt({keys.shape[-1]})) \n"
        )
        print_matrix(attention_weights, tokens, tokens)
        print()

        context_vectors = attention_weights @ values
        print_section_header("Context Vectors", context_vectors.shape)
        print(
            f"Attention Weights @ Values = {tuple(attention_weights.shape)} @ {tuple(values.shape)}\n"
        )
        print_matrix(context_vectors, tokens, ["dim0", "dim1"])
        print()

        return context_vectors


torch.manual_seed(789)
d_in = inputs.shape[1]
d_out = 2
sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v2(inputs)
