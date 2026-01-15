"""Chart generation for evaluation results."""

from pathlib import Path
from typing import List

from openadapt_grounding.eval.metrics.types import MethodMetrics


def generate_comparison_charts(
    metrics: List[MethodMetrics],
    output_dir: Path,
) -> None:
    """Generate comparison charts as PNG files.

    Charts generated:
    1. Overall detection rate bar chart
    2. Detection rate by element size
    3. Latency vs accuracy scatter plot

    Args:
        metrics: List of MethodMetrics from different runs
        output_dir: Directory to save chart images
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping chart generation.")
        print("Install with: uv pip install matplotlib")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics:
        print("No metrics to visualize.")
        return

    # Color palette
    colors = [
        "#2196f3",
        "#4caf50",
        "#ff9800",
        "#f44336",
        "#9c27b0",
        "#607d8b",
        "#00bcd4",
        "#795548",
    ]

    # Chart 1: Overall Detection Rate
    _generate_detection_rate_chart(metrics, output_dir, colors)

    # Chart 2: Detection Rate by Size
    _generate_size_breakdown_chart(metrics, output_dir, colors)

    # Chart 3: Latency vs Accuracy
    _generate_latency_accuracy_chart(metrics, output_dir, colors)

    print(f"Generated 3 charts in {output_dir}")


def _generate_detection_rate_chart(
    metrics: List[MethodMetrics],
    output_dir: Path,
    colors: List[str],
) -> None:
    """Generate overall detection rate bar chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    methods = [f"{m.method_name}\n({m.dataset_name})" for m in metrics]
    rates = [m.detection_rate * 100 for m in metrics]

    bars = ax.bar(
        methods, rates, color=[colors[i % len(colors)] for i in range(len(methods))]
    )
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Overall Detection Rate by Method")
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "detection_rate_overall.png", dpi=150)
    plt.close()


def _generate_size_breakdown_chart(
    metrics: List[MethodMetrics],
    output_dir: Path,
    colors: List[str],
) -> None:
    """Generate detection rate by size chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    methods = [f"{m.method_name[:15]}\n({m.dataset_name})" for m in metrics]
    x = range(len(methods))
    width = 0.25

    small = [m.detection_rate_small * 100 for m in metrics]
    medium = [m.detection_rate_medium * 100 for m in metrics]
    large = [m.detection_rate_large * 100 for m in metrics]

    ax.bar([i - width for i in x], small, width, label="Small (<32px)", color="#f44336")
    ax.bar(x, medium, width, label="Medium (32-100px)", color="#ff9800")
    ax.bar([i + width for i in x], large, width, label="Large (>100px)", color="#4caf50")

    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Detection Rate by Element Size")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "detection_rate_by_size.png", dpi=150)
    plt.close()


def _generate_latency_accuracy_chart(
    metrics: List[MethodMetrics],
    output_dir: Path,
    colors: List[str],
) -> None:
    """Generate latency vs accuracy scatter plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, m in enumerate(metrics):
        ax.scatter(
            m.mean_latency_ms,
            m.detection_rate * 100,
            s=200,
            label=f"{m.method_name} ({m.dataset_name})",
            c=[colors[i % len(colors)]],
        )
        ax.annotate(
            m.method_name[:12],
            (m.mean_latency_ms, m.detection_rate * 100),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Accuracy vs Latency Tradeoff")

    # Add reference lines
    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.3, label="80% target")
    ax.axvline(x=2000, color="gray", linestyle="--", alpha=0.3, label="2s target")

    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_latency.png", dpi=150)
    plt.close()
