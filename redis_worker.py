"""
sqs_worker.py
=============
The background daemon that processes chunks from AWS SQS FIFO.
This worker is stateless; it reads/writes state to Amazon ElastiCache (Valkey).

Usage: python sqs_worker.py
"""

import time
import json
import logging
import os
import requests
from aws_mock import pop_from_redis_queue, get_session_state, set_session_state, acknowledge_message
from stream_processor import process_single_chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("worker")

# The URL for the AWS API Gateway WebSockets or Frontend Webhook
# In production, set this in your EKS environment variables.
FRONTEND_WEBHOOK_URL = os.getenv("FRONTEND_WEBHOOK_URL")

def push_to_frontend(session_id: str, source: str, chunk_turns: list, updated_fields: list):
    """
    Pushes the Delta JSON to the frontend webhook.
    """
    if not updated_fields:
        log.info(f"[{session_id}] No fields were answered/updated in this chunk. Skipping frontend push.")
        return

    # Build the exact JSON payload the frontend expects
    payload = {
        "sessionId": session_id,
        "source": source,
        "conversation": chunk_turns,
        "summary": {
            "questions": updated_fields
        }
    }

    log.info(f"\n{'='*55}\n[WEBHOOK] Pushing updated fields for {session_id}...")
    
    if FRONTEND_WEBHOOK_URL:
        try:
            log.info(f"Sending POST request to {FRONTEND_WEBHOOK_URL}...")
            r = requests.post(FRONTEND_WEBHOOK_URL, json=payload, timeout=10)
            if r.status_code in (200, 201, 202, 204):
                log.info(f"Successfully pushed to frontend! Status: {r.status_code}")
            else:
                log.error(f"Failed to push to frontend. Status: {r.status_code}, Response: {r.text}")
        except Exception as e:
            log.error(f"Error sending webhook to frontend: {e}")
    else:
        # If no URL is configured, just print to terminal for local debugging
        log.warning("No FRONTEND_WEBHOOK_URL configured in environment. Printing to terminal instead:")
        print(json.dumps(payload, indent=2))
        
    log.info(f"{'='*55}\n")

def run_worker():
    log.info("Starting Redis Worker Daemon... Polling for messages...")
    
    # We maintain an internal chunk counter and chunk log per session
    session_chunk_counters = {}
    session_chunk_logs = {}  # accumulates per-chunk detail for the output file
    
    while True:
        try:
            msg_id, msg_data = pop_from_redis_queue()
            if not msg_id:
                # Sleep a bit if queue is empty
                time.sleep(2.0)
                continue
                
            log.info(f"Picked up message {msg_id} from Redis Queue.")
            
            session_id = msg_data.get("sessionId")
            source = msg_data.get("source")
            chunk_turns = msg_data.get("chunk_turns", [])
            
            # Fetch active state from Valkey/Redis
            state = get_session_state(session_id)
            if not state:
                log.error(f"Cannot process {msg_id}: State for session {session_id} not found in Valkey.")
                continue
                
            all_fields = state.get("all_fields", [])
            
            # Update counter
            chunk_num = session_chunk_counters.get(session_id, 0) + 1
            session_chunk_counters[session_id] = chunk_num
            
            # ── Run the core LLM processing logic ──
            # process_single_chunk modifies `all_fields` in place and returns
            # the updated fields + a detailed chunk log entry
            updated_fields, chunk_log_entry = process_single_chunk(
                all_fields, 
                chunk_turns, 
                chunk_num=chunk_num,
                session_id=session_id
            )
            
            # Save the mutated state back to Valkey/Redis
            state["all_fields"] = all_fields
            set_session_state(session_id, state)
            
            # Accumulate chunk log entry
            if session_id not in session_chunk_logs:
                session_chunk_logs[session_id] = []
            session_chunk_logs[session_id].append(chunk_log_entry)

            # Save rich per-chunk output file after every chunk
            outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            output_file = os.path.join(outputs_dir, f"{session_id}_output.json")
            output_data = {
                "sessionId": session_id,
                "chunks": session_chunk_logs[session_id],
                "all_fields_current_state": [
                    {"id": f.get("id"), "label": f.get("label"), "value": f.get("value", "")}
                    for f in all_fields
                ]
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            log.info(f"[{session_id}] Output saved → outputs/{session_id}_output.json")
            
            # Push the updated fields back to the frontend
            push_to_frontend(session_id, source, chunk_turns, updated_fields)
            
            # ✅ Acknowledge the message ONLY after full success.
            # This removes it from the processing queue permanently.
            acknowledge_message(msg_id, msg_data)
            
            log.info(f"Successfully finished processing message {msg_id}.")
            
        except Exception as e:
            log.error(f"Worker encountered an error: {e}", exc_info=True)
            time.sleep(5.0)

if __name__ == "__main__":
    run_worker()
