# Claude Code Instructions

## Deployment Commands

ALWAYS use the CLI commands in `openadapt_grounding.deploy` for deployment operations. NEVER run raw SSH/docker commands directly.

```bash
# Full deployment
python -m openadapt_grounding.deploy start

# Check status
python -m openadapt_grounding.deploy status

# Container operations
python -m openadapt_grounding.deploy ps      # Show container status
python -m openadapt_grounding.deploy logs    # Show container logs (--lines=N)
python -m openadapt_grounding.deploy run     # Start container
python -m openadapt_grounding.deploy build   # Build Docker image
python -m openadapt_grounding.deploy test    # Test endpoint

# Instance operations
python -m openadapt_grounding.deploy ssh     # SSH into instance
python -m openadapt_grounding.deploy stop    # Terminate instance
```

## Adding New Operations

If you need a deployment operation that doesn't exist:
1. Add it as a method to the `Deploy` class in `src/openadapt_grounding/deploy/deploy.py`
2. Update the docstrings in both `deploy.py` and `__main__.py`
3. Update this file with the new command

## Configuration

Edit `src/openadapt_grounding/deploy/config.py` for deployment settings.
Copy `.env.example` to `.env` and fill in AWS credentials.
