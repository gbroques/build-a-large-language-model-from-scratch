import torch
from typing import List, Optional, Tuple

# Formatting constants
LABEL_WIDTH = 8
DECIMAL_PLACES = 3


def print_section_header(title: str, shape: torch.Size) -> None:
    """Print formatted section header with title and shape."""
    heading = f"{title} {tuple(shape)}"
    print(heading)
    print("-" * len(heading))


def print_section(
    title: str,
    data: torch.Tensor,
    row_labels: List[str],
    col_labels: Optional[List[str]] = None,
) -> None:
    """Print formatted tensor data with row and optional column labels."""
    print_section_header(title, data.shape)
    if col_labels:
        print(
            " " * LABEL_WIDTH,
            " ".join(f"{label:>{LABEL_WIDTH}}" for label in col_labels),
        )
    for i, row_label in enumerate(row_labels):
        print(
            f"{row_label:>{LABEL_WIDTH}}",
            " ".join(
                f"{data[i][j]:{LABEL_WIDTH}.{DECIMAL_PLACES}f}"
                for j in range(data.shape[1])
            ),
        )
    print()


# 6 tokens, with an embedding dimension of 3.
tokens = ["Your", "journey", "starts", "with", "one", "step"]
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

print_section("Token Embeddings", inputs, tokens)

# Compute the dot product between all elements to determine which elements "attend to" each other.
# The higher the dot product, the higher the similarity and attention score between two elements.
#
# Shape: 6x3 @ 3x6 = 6x6
#
# attention_scores[i][j] represents how much token i attends to token j.
#
# For example:
# - attention_scores[0][2] = how much "Your" (token 0) attends to "starts" (token 2)
# - attention_scores[3][1] = how much "with" (token 3) attends to "journey" (token 1)
#
# The diagonal elements attention_scores[i][i] represent how much each token attends to itself.
#
attention_scores = inputs @ inputs.T
print_section("Attention Scores", attention_scores, tokens, tokens)

# Set dim=-1 normalize across the columns so that the values in each row sum up to 1
attention_weights = torch.softmax(attention_scores, dim=-1)
print_section("Attention Weights", attention_weights, tokens, tokens)
