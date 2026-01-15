"""CLI entry point for eval module.

Usage:
    python -m openadapt_grounding.eval generate --type synthetic --count 500
    python -m openadapt_grounding.eval run --method omniparser --dataset synthetic
    python -m openadapt_grounding.eval run --method uitars --dataset synthetic
    python -m openadapt_grounding.eval compare --charts-dir evaluation/charts
"""

from openadapt_grounding.eval.cli import main

if __name__ == "__main__":
    main()
