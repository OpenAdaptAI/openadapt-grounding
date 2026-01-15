"""CLI for evaluation framework.

Usage:
    python -m openadapt_grounding.eval generate --type synthetic --count 500
    python -m openadapt_grounding.eval run --method omniparser --dataset synthetic
    python -m openadapt_grounding.eval run --method uitars --dataset synthetic
    python -m openadapt_grounding.eval compare --charts-dir evaluation/charts
"""

import json
from pathlib import Path
from typing import List, Optional

try:
    import fire
except ImportError:
    raise ImportError("fire not installed. Run: uv pip install fire")

from PIL import Image

from openadapt_grounding.eval.config import get_settings
from openadapt_grounding.eval.dataset.schema import Dataset
from openadapt_grounding.eval.dataset.synthetic import SyntheticGenerator
from openadapt_grounding.eval.methods.base import EvaluationMethod
from openadapt_grounding.eval.methods.cropping import (
    FixedCropping,
    NoCropping,
    ScreenSeekeRCropping,
)
from openadapt_grounding.eval.metrics.compute import aggregate_metrics
from openadapt_grounding.eval.metrics.types import ElementResult, MethodMetrics
from openadapt_grounding.eval.results.compare import compare_methods
from openadapt_grounding.eval.results.storage import load_results, save_results
from openadapt_grounding.eval.visualization.charts import generate_comparison_charts
from openadapt_grounding.eval.visualization.tables import print_summary_table


class EvalCLI:
    """Evaluation CLI for openadapt-grounding.

    Commands:
        generate    Generate an evaluation dataset
        run         Run evaluation for a method on a dataset
        compare     Compare all evaluated methods
        list        List available methods and datasets
    """

    def generate(
        self,
        type: str = "synthetic",
        count: int = 500,
        output_dir: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        """Generate an evaluation dataset.

        Args:
            type: Dataset type: "synthetic"
            count: Number of samples to generate
            output_dir: Directory to save dataset (default: evaluation/datasets/{type})
            seed: Random seed for reproducibility
        """
        settings = get_settings()
        if output_dir is None:
            output_dir = settings.DATASETS_DIR

        output_path = Path(output_dir) / type

        if type == "synthetic":
            print(f"Generating {count} synthetic samples...")
            generator = SyntheticGenerator(
                width=settings.SYNTHETIC_WIDTH,
                height=settings.SYNTHETIC_HEIGHT,
                seed=seed,
            )
            dataset = generator.generate_dataset(output_path, num_samples=count)
            print(f"Generated {len(dataset.samples)} samples")
            print(f"Total elements: {dataset.total_elements()}")
            print(f"Saved to: {output_path}")

        else:
            print(f"Unknown dataset type: {type}")
            print("Available types: synthetic")

    def run(
        self,
        method: str,
        dataset: str,
        dataset_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        omniparser_url: Optional[str] = None,
        uitars_url: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        """Run evaluation for a specific method on a dataset.

        Args:
            method: Method name: "omniparser", "omniparser-fixed", "omniparser-screenseeker",
                    "uitars", "uitars-fixed", "uitars-screenseeker"
            dataset: Dataset name: "synthetic"
            dataset_dir: Base directory for datasets
            output_dir: Directory to save results
            omniparser_url: OmniParser server URL
            uitars_url: UI-TARS server URL
            limit: Limit number of samples to evaluate (for testing)
        """
        settings = get_settings()
        dataset_dir = dataset_dir or settings.DATASETS_DIR
        output_dir = output_dir or settings.RESULTS_DIR
        omniparser_url = omniparser_url or settings.OMNIPARSER_URL
        uitars_url = uitars_url or settings.UITARS_URL

        # Load dataset
        dataset_path = Path(dataset_dir) / dataset / "annotations.json"
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            print(f"Run 'python -m openadapt_grounding.eval generate --type {dataset}' first")
            return

        eval_dataset = Dataset.load(dataset_path)
        print(f"Loaded {len(eval_dataset.samples)} samples from {dataset}")
        print(f"Total elements: {eval_dataset.total_elements()}")

        # Create method
        eval_method = self._create_method(method, omniparser_url, uitars_url)
        if eval_method is None:
            return

        # Check availability
        if not eval_method.is_available():
            print(f"Method backend not available: {eval_method.name}")
            print("Please ensure the server is running.")
            return

        print(f"Running evaluation: {eval_method.name} on {dataset}")

        # Run evaluation
        results = self._run_evaluation(
            eval_method, eval_dataset, Path(dataset_dir) / dataset, limit
        )

        # Compute metrics
        metrics = aggregate_metrics(results, eval_method.name, dataset)

        # Save results
        output_path = Path(output_dir) / f"{method}_{dataset}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results(metrics, output_path)

        # Print summary
        print_summary_table([metrics])
        print(f"Results saved to: {output_path}")

    def compare(
        self,
        results_dir: Optional[str] = None,
        output: Optional[str] = None,
        charts_dir: Optional[str] = None,
    ) -> None:
        """Compare all evaluated methods and generate visualizations.

        Args:
            results_dir: Directory containing individual result files
            output: Path for comparison JSON output
            charts_dir: Directory to save visualization charts
        """
        settings = get_settings()
        results_dir = results_dir or settings.RESULTS_DIR
        charts_dir = charts_dir or settings.CHARTS_DIR

        results_path = Path(results_dir)
        result_files = list(results_path.glob("*.json"))

        if not result_files:
            print(f"No results found in {results_dir}")
            print("Run 'python -m openadapt_grounding.eval run --method <method> --dataset <dataset>' first")
            return

        print(f"Found {len(result_files)} result files")

        # Load all results
        all_metrics: List[MethodMetrics] = []
        for f in result_files:
            try:
                metrics = load_results(f)
                all_metrics.append(metrics)
                print(f"  Loaded: {f.name}")
            except Exception as e:
                print(f"  Error loading {f.name}: {e}")

        if not all_metrics:
            print("No valid results to compare.")
            return

        # Print comparison table
        print_summary_table(all_metrics)

        # Generate comparison
        comparison = compare_methods(all_metrics)

        # Save comparison
        if output:
            output_path = Path(output)
        else:
            output_path = results_path / "comparison.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to: {output_path}")

        # Generate charts
        if charts_dir:
            charts_path = Path(charts_dir)
            generate_comparison_charts(all_metrics, charts_path)

    def list(self) -> None:
        """List available methods and datasets."""
        print("\nAvailable Methods:")
        print("-" * 40)
        methods = [
            ("omniparser", "OmniParser baseline"),
            ("omniparser-fixed", "OmniParser + fixed cropping"),
            ("omniparser-screenseeker", "OmniParser + ScreenSeekeR cropping"),
            ("uitars", "UI-TARS baseline"),
            ("uitars-fixed", "UI-TARS + fixed cropping"),
            ("uitars-screenseeker", "UI-TARS + ScreenSeekeR cropping"),
        ]
        for name, desc in methods:
            print(f"  {name:<25} {desc}")

        print("\nDataset Types:")
        print("-" * 40)
        datasets = [
            ("synthetic", "Auto-generated UI screenshots"),
        ]
        for name, desc in datasets:
            print(f"  {name:<25} {desc}")

        settings = get_settings()
        print(f"\nDefault Directories:")
        print(f"  Datasets: {settings.DATASETS_DIR}")
        print(f"  Results:  {settings.RESULTS_DIR}")
        print(f"  Charts:   {settings.CHARTS_DIR}")

    def _create_method(
        self, method_name: str, omniparser_url: str, uitars_url: str
    ) -> Optional[EvaluationMethod]:
        """Create evaluation method from name."""
        # Parse method name
        parts = method_name.lower().split("-")
        base = parts[0]
        cropping = parts[1] if len(parts) > 1 else "baseline"

        # Create cropping strategy
        if cropping == "baseline":
            crop_strategy = NoCropping()
        elif cropping == "fixed":
            crop_strategy = FixedCropping()
        elif cropping == "screenseeker":
            crop_strategy = ScreenSeekeRCropping()
        else:
            print(f"Unknown cropping strategy: {cropping}")
            print("Available: baseline, fixed, screenseeker")
            return None

        # Create method
        if base == "omniparser":
            from openadapt_grounding.parsers.omniparser import OmniParserClient
            from openadapt_grounding.eval.methods.omniparser import OmniParserMethod

            client = OmniParserClient(omniparser_url)
            return OmniParserMethod(client, crop_strategy)

        elif base == "uitars":
            from openadapt_grounding.parsers.uitars import UITarsClient
            from openadapt_grounding.eval.methods.uitars import UITarsMethod

            client = UITarsClient(uitars_url)
            return UITarsMethod(client, crop_strategy)

        else:
            print(f"Unknown base method: {base}")
            print("Available: omniparser, uitars")
            return None

    def _run_evaluation(
        self,
        method: EvaluationMethod,
        dataset: Dataset,
        dataset_path: Path,
        limit: Optional[int] = None,
    ) -> List[ElementResult]:
        """Run evaluation and return results."""
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            use_tqdm = False

        from openadapt_grounding.eval.metrics.compute import compute_iou

        results: List[ElementResult] = []

        samples = dataset.samples
        if limit:
            samples = samples[:limit]

        iterator = tqdm(samples, desc="Evaluating") if use_tqdm else samples

        for sample in iterator:
            image_path = dataset_path / sample.image_path
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue

            image = Image.open(image_path)

            for element in sample.elements:
                prediction = method.evaluate_element(image, element)

                # Compute IoU if we have bbox predictions
                iou = 0.0
                if prediction.bbox is not None:
                    iou = compute_iou(prediction.bbox, element.bbox)

                # Compute distance from predicted point to target center
                distance = 0.0
                if prediction.click_point is not None:
                    target_cx, target_cy = element.click_point
                    pred_cx, pred_cy = prediction.click_point
                    distance = (
                        (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
                    ) ** 0.5

                results.append(
                    ElementResult(
                        sample_id=sample.id,
                        element_id=element.id,
                        found=prediction.found,
                        iou=iou,
                        latency_ms=prediction.latency_ms,
                        attempts=prediction.attempts,
                        size_category=element.size_category.value,
                        element_type=element.element_type.value,
                        distance_from_target=distance,
                        method_info=prediction.method_info,
                    )
                )

        return results


def main():
    """Entry point for CLI."""
    fire.Fire(EvalCLI)


if __name__ == "__main__":
    main()
