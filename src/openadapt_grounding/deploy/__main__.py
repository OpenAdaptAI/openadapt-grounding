"""CLI entry point for deploy module.

Usage:
    python -m openadapt_grounding.deploy start
    python -m openadapt_grounding.deploy status
    python -m openadapt_grounding.deploy ssh
    python -m openadapt_grounding.deploy stop
"""

from openadapt_grounding.deploy.deploy import main

if __name__ == "__main__":
    main()
