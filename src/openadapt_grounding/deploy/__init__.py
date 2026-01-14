"""OmniParser deployment utilities.

Deploy OmniParser v2 to AWS EC2 with GPU support.

Usage:
    # From command line
    python -m openadapt_grounding.deploy start
    python -m openadapt_grounding.deploy status
    python -m openadapt_grounding.deploy stop

    # From Python
    from openadapt_grounding.deploy import Deploy, settings
    Deploy.start()
"""

from openadapt_grounding.deploy.config import settings
from openadapt_grounding.deploy.deploy import Deploy

__all__ = ["Deploy", "settings"]
