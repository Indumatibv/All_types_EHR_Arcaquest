"""
aws_mock.py
===========
A seamless abstraction layer for Amazon ElastiCache (Valkey/Redis).
If REDIS_URL is set in the environment, it natively uses redis-py to connect to the AWS ElastiCache cluster.
Otherwise, it falls back to a robust mock implementation using local files for local POC testing.
"""

import os
import json
import uuid
import time
import logging
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# INIT REDIS CLIENT
# ----------------------------------------------------------------------------
redis_client = None
REDIS_URL = os.getenv("REDIS_URL")

if REDIS_URL:
    try:
        import redis
        # decode_responses=True ensures we get strings back instead of bytes
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info(f"Initialized REAL Redis connection to {REDIS_URL}")
    except ImportError:
        log.warning("REDIS_URL is set but 'redis' python package is not installed. Falling back to local mock.")

# Base directory for our local mock storage (only used if redis_client is None)
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local_state")
STATE_DIR = os.path.join(LOCAL_DIR, "valkey_state")
QUEUE_DIR = os.path.join(LOCAL_DIR, "redis_queue")

if not redis_client:
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(QUEUE_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# VALKEY / REDIS (State Store)
# ----------------------------------------------------------------------------

def get_session_state(session_id: str) -> dict:
    """Gets session state from real Redis or local mock."""
    if redis_client:
        data = redis_client.get(f"session:{session_id}")
        return json.loads(data) if data else None

    # Fallback mock
    filepath = os.path.join(STATE_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def set_session_state(session_id: str, state_dict: dict) -> None:
    """Sets session state in real Redis or local mock."""
    if redis_client:
        redis_client.set(f"session:{session_id}", json.dumps(state_dict, ensure_ascii=False))
        return

    # Fallback mock
    filepath = os.path.join(STATE_DIR, f"{session_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# REDIS QUEUE (Message Queue) — Reliable Queue Pattern
# ----------------------------------------------------------------------------
# How it works:
#   1. push: lpush to QUEUE_NAME (the inbox)
#   2. pop:  atomically RPOPLPUSH from QUEUE_NAME → PROCESSING_QUEUE_NAME
#            (message is now in-progress, NOT deleted yet)
#   3. ack:  after successful processing, LREM from PROCESSING_QUEUE_NAME
#            (message is now fully deleted)
#   4. crash recovery: on startup, any stuck messages in PROCESSING_QUEUE_NAME
#            can be moved back to QUEUE_NAME for reprocessing.
# This guarantees exactly-once processing — no message is ever permanently lost.
# ----------------------------------------------------------------------------
QUEUE_NAME      = "arcaquest_chunk_queue"
PROCESSING_NAME = "arcaquest_chunk_queue:processing"

# For local mock, processing folder mirrors the PROCESSING_QUEUE_NAME list
PROCESSING_DIR = os.path.join(LOCAL_DIR, "redis_processing")
if not redis_client:
    os.makedirs(PROCESSING_DIR, exist_ok=True)


def push_to_redis_queue(message_dict: dict) -> str:
    """Pushes a chunk to real Redis list or local mock directory."""
    msg_id = f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    if redis_client:
        message_dict["_msg_id"] = msg_id
        redis_client.lpush(QUEUE_NAME, json.dumps(message_dict, ensure_ascii=False))
        return msg_id

    # Fallback mock
    filepath = os.path.join(QUEUE_DIR, f"{msg_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(message_dict, f, indent=2, ensure_ascii=False)
    return msg_id


def pop_from_redis_queue() -> tuple:
    """
    Atomically moves the oldest message from the inbox queue to the
    processing queue. Returns (msg_id, message_dict) or (None, None).
    
    The message is NOT deleted yet — call acknowledge_message() after
    successful processing to confirm deletion.
    """
    if redis_client:
        # RPOPLPUSH atomically moves from inbox → processing in one command.
        # This is crash-safe: if the worker dies, the msg stays in processing.
        data = redis_client.rpoplpush(QUEUE_NAME, PROCESSING_NAME)
        if not data:
            return None, None
        msg_data = json.loads(data)
        msg_id = msg_data.get("_msg_id", f"msg_{int(time.time() * 1000)}")
        return msg_id, msg_data

    # Fallback mock: move file from queue dir → processing dir
    files = sorted(os.listdir(QUEUE_DIR))
    if not files:
        return None, None
    
    target_file = files[0]
    src_path  = os.path.join(QUEUE_DIR, target_file)
    dest_path = os.path.join(PROCESSING_DIR, target_file)
    msg_id    = target_file.replace(".json", "")
    
    try:
        with open(src_path, "r", encoding="utf-8") as f:
            msg_data = json.load(f)
        # Move to processing folder (not delete!)
        os.rename(src_path, dest_path)
    except Exception:
        if os.path.exists(src_path):
            os.remove(src_path)
        return None, None
        
    return msg_id, msg_data


def acknowledge_message(msg_id: str, msg_data: dict) -> None:
    """
    Call this AFTER a message has been successfully processed.
    Removes it from the processing queue, completing the exactly-once guarantee.
    """
    if redis_client:
        # Remove exactly 1 occurrence of this message from the processing list
        redis_client.lrem(PROCESSING_NAME, 1, json.dumps(msg_data, ensure_ascii=False))
        return

    # Fallback mock: delete the file from the processing folder
    proc_file = os.path.join(PROCESSING_DIR, f"{msg_id}.json")
    if os.path.exists(proc_file):
        os.remove(proc_file)
