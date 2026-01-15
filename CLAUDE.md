# Claude Code Instructions

## CRITICAL: Non-Interactive Operations

**ALL operations MUST be fully programmatic with NO interactive prompts.**

- SSH commands must use `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`
- Never use `tail -f` or other blocking commands that require Ctrl+C
- Use `timeout` wrapper or `--timeout` flags for potentially long operations
- Run long operations in background and poll for status
- All CLI tools must accept all required input via arguments/flags

## Deployment Commands

ALWAYS use the CLI commands in `openadapt_grounding.deploy` for deployment operations. NEVER run raw SSH/docker commands directly.

Use `uv run` to execute commands:

### OmniParser Deployment

```bash
# Full deployment
uv run python -m openadapt_grounding.deploy start

# Check status
uv run python -m openadapt_grounding.deploy status

# Container operations
uv run python -m openadapt_grounding.deploy ps      # Show container status
uv run python -m openadapt_grounding.deploy logs    # Show container logs (--lines=N)
uv run python -m openadapt_grounding.deploy run     # Start container
uv run python -m openadapt_grounding.deploy build   # Build Docker image
uv run python -m openadapt_grounding.deploy test    # Test endpoint

# Instance operations
uv run python -m openadapt_grounding.deploy ssh     # SSH into instance
uv run python -m openadapt_grounding.deploy stop    # Terminate instance
```

### UI-TARS Deployment

UI-TARS is deployed separately on its own instance using vLLM.

```bash
# Full deployment
uv run python -m openadapt_grounding.deploy.uitars start

# Check status
uv run python -m openadapt_grounding.deploy.uitars status

# Container operations
uv run python -m openadapt_grounding.deploy.uitars ps      # Show container status
uv run python -m openadapt_grounding.deploy.uitars logs    # Show container logs (--lines=N)
uv run python -m openadapt_grounding.deploy.uitars run     # Start container
uv run python -m openadapt_grounding.deploy.uitars build   # Build Docker image
uv run python -m openadapt_grounding.deploy.uitars test    # Test grounding endpoint

# Instance operations
uv run python -m openadapt_grounding.deploy.uitars ssh     # SSH into instance
uv run python -m openadapt_grounding.deploy.uitars stop    # Terminate instance
```

## Adding New Operations

If you need a deployment operation that doesn't exist:
1. Add it as a method to the `Deploy` class in the relevant deploy.py:
   - OmniParser: `src/openadapt_grounding/deploy/deploy.py`
   - UI-TARS: `src/openadapt_grounding/deploy/uitars/deploy.py`
2. Update the docstrings in both `deploy.py` and `__main__.py`
3. Update this file with the new command

## Configuration

Edit `src/openadapt_grounding/deploy/config.py` for deployment settings:
- `DeploySettings` - OmniParser configuration
- `UITarsSettings` - UI-TARS configuration

Copy `.env.example` to `.env` and fill in AWS credentials.
