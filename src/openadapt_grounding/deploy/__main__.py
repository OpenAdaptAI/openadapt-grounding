"""CLI entry point for deploy module.

Usage:
    python -m openadapt_grounding.deploy start   # Deploy new instance
    python -m openadapt_grounding.deploy status  # Check status
    python -m openadapt_grounding.deploy ssh     # SSH into instance
    python -m openadapt_grounding.deploy stop    # Terminate instance
    python -m openadapt_grounding.deploy logs    # Show container logs
    python -m openadapt_grounding.deploy ps      # Show container status
    python -m openadapt_grounding.deploy build   # Build Docker image
    python -m openadapt_grounding.deploy run     # Start container
    python -m openadapt_grounding.deploy test    # Test endpoint
"""

from openadapt_grounding.deploy.deploy import main

if __name__ == "__main__":
    main()
