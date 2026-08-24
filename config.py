"""
config.py
=========

Handles dynamic model configuration loading.
Allows changing the active LLM or Reranker models without redeploying code.

Resolution Order:
  1. AWS SSM Parameter Store (if running in AWS / permissions allow)
  2. Environment Variables (.env)
  3. Hardcoded defaults (fallback)
"""

import os
import time
import logging
from dotenv import load_dotenv

# Load .env variables first
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

log = logging.getLogger(__name__)

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    log.debug("boto3 not installed. AWS SSM Parameter Store fetching will be disabled.")

# --- SSM Parameter Names ---
SSM_NOVA_MODEL_PARAM = os.getenv("SSM_NOVA_MODEL_PARAM", "/arcaquest/prod/nova_model_id")
SSM_RERANKER_MODEL_PARAM = os.getenv("SSM_RERANKER_MODEL_PARAM", "/arcaquest/prod/reranker_model_id")
SSM_RERANKER_URL_PARAM = os.getenv("SSM_RERANKER_URL_PARAM", "/arcaquest/prod/reranker_url")

# --- Local Defaults (from .env or hardcoded) ---
DEFAULT_NOVA_MODEL = os.getenv("NOVA_MODEL", "us.amazon.nova-pro-v1:0")
DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "danielchalef/Qwen3-Reranker-4B-seq-cls-vllm-fixed")
DEFAULT_RERANKER_URL = os.getenv("RERANKER_URL", "https://94u5v92s73.execute-api.ap-south-1.amazonaws.com/dev/v1/rerank")

# --- Cache ---
_config_cache = {}
_cache_expiry = 0
CACHE_TTL = 300  # 5 minutes


def _fetch_from_ssm(param_name: str, default_val: str) -> str:
    """Attempt to fetch a parameter from AWS SSM."""
    if not BOTO3_AVAILABLE:
        return default_val
        
    try:
        ssm = boto3.client("ssm", region_name=DEFAULT_REGION)
        response = ssm.get_parameter(Name=param_name, WithDecryption=False)
        return response["Parameter"]["Value"]
    except Exception as e:
        log.debug(f"Failed to fetch {param_name} from SSM, using fallback. Reason: {e}")
        return default_val


def get_model_config() -> dict:
    """
    Returns the current active model configurations.
    Uses an in-memory cache for 5 minutes to avoid excessive SSM calls.
    """
    global _config_cache, _cache_expiry

    now = time.time()
    if _config_cache and now < _cache_expiry:
        return _config_cache

    log.info("Refreshing dynamic model configuration...")
    
    # In a real production deployment with SSM enabled, these would fetch from AWS.
    # If AWS credentials or SSM permissions are absent, it safely falls back to defaults.
    nova_model = _fetch_from_ssm(SSM_NOVA_MODEL_PARAM, DEFAULT_NOVA_MODEL)
    reranker_model = _fetch_from_ssm(SSM_RERANKER_MODEL_PARAM, DEFAULT_RERANKER_MODEL)
    reranker_url = _fetch_from_ssm(SSM_RERANKER_URL_PARAM, DEFAULT_RERANKER_URL)

    _config_cache = {
        "nova_model": nova_model,
        "region": DEFAULT_REGION,
        "reranker_model": reranker_model,
        "reranker_url": reranker_url,
    }
    _cache_expiry = now + CACHE_TTL

    log.info(f"Active LLM: {nova_model}")
    log.info(f"Active Reranker: {reranker_model}")
    return _config_cache
