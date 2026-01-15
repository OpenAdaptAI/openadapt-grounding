"""CLI entry point for UI-TARS deployment.

Usage:
    python -m openadapt_grounding.deploy.uitars start   # Deploy new instance
    python -m openadapt_grounding.deploy.uitars status  # Check status
    python -m openadapt_grounding.deploy.uitars ssh     # SSH into instance
    python -m openadapt_grounding.deploy.uitars stop    # Terminate instance
    python -m openadapt_grounding.deploy.uitars logs    # Show container logs
    python -m openadapt_grounding.deploy.uitars ps      # Show container status
    python -m openadapt_grounding.deploy.uitars build   # Build Docker image
    python -m openadapt_grounding.deploy.uitars run     # Start container
    python -m openadapt_grounding.deploy.uitars test    # Test endpoint
"""

from .deploy import main

if __name__ == "__main__":
    main()
