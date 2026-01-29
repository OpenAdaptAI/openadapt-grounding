# CHANGELOG


## v0.1.2 (2026-01-29)

### Bug Fixes

- Use filename-based GitHub Actions badge URL
  ([#2](https://github.com/OpenAdaptAI/openadapt-grounding/pull/2),
  [`0d58eee`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/0d58eee7dd0bbeb97ab485d74e937a438b72333e))

The workflow-name-based badge URL was showing "no status" because GitHub requires workflow runs on
  the specified branch. Using the filename-based URL format
  (actions/workflows/publish.yml/badge.svg) is more reliable and works regardless of when the
  workflow last ran.

Co-authored-by: Claude Sonnet 4.5 <noreply@anthropic.com>


## v0.1.1 (2026-01-29)

### Bug Fixes

- **ci**: Remove build_command from semantic-release config
  ([`db01f86`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/db01f86bb0e62df393b93c297374325169f893c0))

The python-semantic-release action runs in a Docker container where uv is not available. Let the
  workflow handle building instead.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Continuous Integration

- Switch to python-semantic-release for automated versioning
  ([`a2dd7d5`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/a2dd7d5f8dde359396b5c81515f38bbca3c7a33b))

Replaces manual tag-triggered publish with python-semantic-release: - Automatic version bumping
  based on conventional commits - feat: -> minor, fix:/perf: -> patch - Creates GitHub releases
  automatically - Publishes to PyPI on release

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.1.0 (2026-01-16)

### Bug Fixes

- **uitars**: Fix Dockerfile for vLLM deployment
  ([`5d457ed`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/5d457ed464bb13e10d0fbe8d57a23f85a8f6a31a))

- Fix CMD format: vLLM image has ENTRYPOINT, CMD should be args only - Fix --limit-mm-per-prompt
  format: use KEY=VALUE instead of JSON - Reduce max-model-len from 32768 to 8192 to fix CUDA OOM on
  L4 24GB - Remove model pre-download (causes disk space issues, download at runtime) - Increase
  health check start-period to 600s for model download

Also adds CLI commands: - cleanup: docker system prune for disk space recovery - wait: poll for
  server health with configurable timeout - setup_autoshutdown: create CloudWatch/Lambda
  infrastructure - build --clean: option to cleanup before building - logs: fix stderr capture

Updates CLAUDE.md with non-interactive operations requirements.

### Chores

- Prepare for PyPI publishing and update Gemini models
  ([`7be8b1f`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/7be8b1f734453ab390cf8188d55ce5cdb9932272))

- Add PyPI metadata: maintainers, classifiers, keywords, project URLs - Create LICENSE file (MIT) -
  Add GitHub workflow for trusted PyPI publishing - Update Google provider to use current Gemini
  models (3.x, 2.5.x) - Remove deprecated models (2.0, 1.5) that are retired or retiring - Update
  tests to reflect new model names

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Documentation

- Add literature review, experiment plan, and evaluation harness
  ([`13e69f8`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/13e69f8d3d74dbc5eafe72affd46ce61026c86fe))

- Add literature review: UI-TARS (61.6%), OmniParser (39.6%), ScreenSeekeR (+254%) - Add experiment
  plan: 6 methods comparison across 3 datasets - Add evaluation harness with metrics and dataset
  formats - Update README with documentation links - Add test assets from OmniParser deployment -
  Fix Dockerfile for Conda ToS and PaddleOCR compatibility - Add deploy CLI commands: logs, ps,
  build, run, test

- Move outdated robust_detection to legacy, update evaluation format
  ([`2f32d31`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/2f32d3196212d5cb5c0aaf70cb93190bed0e2205))

- Move robust_detection.md to docs/legacy/ (superseded by ScreenSeekeR approach) - Update
  evaluation.md Section 6-7 to align with new experiment plan - Compare OmniParser vs UI-TARS
  instead of baseline vs robust transforms

- **readme**: Add CLI usage examples with output
  ([`4d5f392`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/4d5f392e36d14cf7d47dc6139353e93667a0791b))

- Add status, ps, logs command output examples - Show deploy workflow with .env setup - Document all
  available commands

- **readme**: Use uv sync for dev setup
  ([`8087ca5`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/8087ca5a17b3cf9a240fb2886390e41e2b1b5571))

### Features

- Add UI-TARS deployment and client
  ([`da67e66`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/da67e66a789f1c0a6a0bfd8117068fd2921f48b5))

- Add UITarsSettings config class for UI-TARS deployment - Create deploy/uitars module with
  vLLM-based Dockerfile - Implement UITarsClient for grounding via OpenAI-compatible API - Add
  GroundingResult dataclass with coordinate conversion - Include smart_resize() for Qwen2.5-VL
  coordinate scaling - Add [uitars] optional dependency group (openai) - Update CLAUDE.md with
  UI-TARS CLI commands - Update README.md with usage examples and API docs - Add
  uitars_deployment_design.md with full design spec

- **deploy**: Add auto-shutdown and fix PaddleOCR compatibility
  ([`ab758df`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/ab758df747ca1f1924787368b58dce3a5de66655))

- pin PaddleOCR to v2.8.1 for API compatibility - add auto-shutdown for cost management - add
  config.py and .env.example

- **deploy**: Add CLI commands and fix transformers version
  ([`9e6398b`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/9e6398b569918f40356d6c25bdf32ec50b2b3688))

- add logs, ps, build, run, test CLI commands - add CLAUDE.md with deployment instructions - pin
  transformers==4.44.2 for Florence-2 compatibility

- **eval**: Add evaluation framework for comparing grounding methods
  ([`064a431`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/064a4314cef63bd778ef419d4cd646e4afbe6c93))

Implement comprehensive evaluation framework:

- Dataset schema with AnnotatedElement, Sample, Dataset classes - Synthetic UI dataset generator
  (buttons, icons, text, links) - Evaluation methods for OmniParser and UI-TARS - Cropping
  strategies: baseline, fixed, ScreenSeekeR-style - Metrics: detection rate, IoU, latency by
  size/type - Results storage and multi-method comparison - Visualization: charts (matplotlib) and
  console tables - CLI: generate, run, compare, list commands

Usage: python -m openadapt_grounding.eval generate --type synthetic --count 500 python -m
  openadapt_grounding.eval run --method omniparser --dataset synthetic python -m
  openadapt_grounding.eval compare --charts-dir evaluation/charts

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **eval**: Add synthetic_hard evaluation dataset
  ([`c97cf70`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/c97cf70e060ddafcf178ab00169d3d2ee30db52f))

Add a more challenging synthetic evaluation dataset with 48 samples for testing VLM API providers
  and grounding methods. Contains synthetic UI screenshots with annotations for element localization
  testing.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **providers**: Add VLM API providers for Claude, GPT, and Gemini
  ([`9a766d1`](https://github.com/OpenAdaptAI/openadapt-grounding/commit/9a766d1d92c4312b917406f61d65a8139180226e))

Add a unified provider abstraction for Visual Language Model APIs: - Base provider class with
  coordinate normalization and response parsing - Anthropic provider for Claude models
  (claude-sonnet-4-20250514) - OpenAI provider for GPT models (gpt-4o) - Google provider for Gemini
  models (gemini-2.0-flash-exp)

Features: - Lazy loading with optional dependencies per provider - Factory function get_provider()
  with name aliases - Coordinate extraction from model responses with regex fallback - Image
  encoding utilities (base64 conversion) - Comprehensive test suite with mocking

Optional dependencies added to pyproject.toml: - providers-anthropic, providers-openai,
  providers-google - providers (all providers combined)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
