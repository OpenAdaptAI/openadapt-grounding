"""Console and markdown table formatting for evaluation results."""

from typing import List

from openadapt_grounding.eval.metrics.types import MethodMetrics


def print_summary_table(metrics: List[MethodMetrics]) -> None:
    """Print a formatted comparison table to console.

    Args:
        metrics: List of MethodMetrics to display
    """
    if not metrics:
        print("No results to display.")
        return

    # Header
    print("\n" + "=" * 90)
    print("EVALUATION RESULTS")
    print("=" * 90)

    # Column headers
    headers = ["Method", "Dataset", "Det.Rate", "IoU", "Latency", "Attempts", "Total"]
    col_widths = [28, 12, 10, 8, 10, 8, 8]

    header_row = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * 90)

    # Data rows
    for m in metrics:
        row = [
            m.method_name[:27],
            m.dataset_name[:11],
            f"{m.detection_rate * 100:.1f}%",
            f"{m.mean_iou:.3f}",
            f"{m.mean_latency_ms:.0f}ms",
            f"{m.mean_attempts:.1f}",
            str(m.total_elements),
        ]
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

    print("=" * 90)

    # Size breakdown
    print("\nDetection Rate by Element Size:")
    print("-" * 70)
    print(f"{'Method':<28} {'Small':<12} {'Medium':<12} {'Large':<12}")
    print("-" * 70)

    for m in metrics:
        print(
            f"{m.method_name[:27]:<28} "
            f"{m.detection_rate_small * 100:>8.1f}%   "
            f"{m.detection_rate_medium * 100:>8.1f}%   "
            f"{m.detection_rate_large * 100:>8.1f}%"
        )

    print()


def format_markdown_table(metrics: List[MethodMetrics]) -> str:
    """Format results as a markdown table.

    Args:
        metrics: List of MethodMetrics to format

    Returns:
        Markdown table string
    """
    if not metrics:
        return "No results available."

    lines = [
        "| Method | Dataset | Detection Rate | Mean IoU | Latency (ms) | Attempts |",
        "|--------|---------|----------------|----------|--------------|----------|",
    ]

    for m in metrics:
        lines.append(
            f"| {m.method_name} | {m.dataset_name} | "
            f"{m.detection_rate * 100:.1f}% | {m.mean_iou:.3f} | "
            f"{m.mean_latency_ms:.0f} | {m.mean_attempts:.1f} |"
        )

    return "\n".join(lines)
