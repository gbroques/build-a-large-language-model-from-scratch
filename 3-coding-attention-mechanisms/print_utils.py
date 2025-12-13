import torch
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
    calculation: Optional[str] = None,
) -> None:
    """Print formatted tensor data with row and optional column labels."""
    print_section_header(title, data.shape)
    if calculation:
        print(f"{calculation}\n")
    print_matrix(data, row_labels, col_labels, precision)
    print()
