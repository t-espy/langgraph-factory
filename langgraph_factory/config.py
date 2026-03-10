"""Shared configuration for langgraph-factory pipelines."""

import os

# Docker Model Runner (OpenAI-compatible endpoint)
DMR_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/engines/v1")
DMR_API_KEY = os.environ.get("DMR_API_KEY", "local-dummy")

# Models
FOREMAN_MODEL = os.environ.get(
    "FOREMAN_MODEL", "docker.io/ai/deepseek-r1-distill-llama:70B-Q4_K_M"
)
CODER_MODEL = os.environ.get("CODER_MODEL", "docker.io/ai/qwen3-coder-next:latest")

# Output
OUTPUT_DIR = os.environ.get("FACTORY_OUTPUT_DIR", os.path.join(os.getcwd(), "factory_out"))

# Retry limits
MAX_GENERATE_ATTEMPTS = int(os.environ.get("MAX_GENERATE_ATTEMPTS", "2"))
MAX_FIX_ATTEMPTS = int(os.environ.get("MAX_FIX_ATTEMPTS", "4"))
LINT_MAX_LOOPS = int(os.environ.get("LINT_MAX_LOOPS", "10"))
