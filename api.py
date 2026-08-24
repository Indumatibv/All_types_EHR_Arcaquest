"""
api.py
======
The stateless Ingestion API for the ArcaQuest AWS Stream Processing.
Built with Flask.

Endpoints:
  - POST /session/init
  - POST /session/chunk
"""

import logging
from flask import Flask, request, jsonify

# In a real AWS environment, you'd use boto3/redis-py here.
# For local POC testing, we use our local mock layer:
from aws_mock import set_session_state, get_session_state, push_to_redis_queue
from field_handlers import flatten_all_fields

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("api")

app = Flask(__name__)

@app.route("/session/init", methods=["POST"])
def session_init():
    """
    Initializes a new streaming session.
    Expects payload: { "sessionId": "...", "source": "...", "summary": { "questions": [...] } }
    """
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    session_id = data.get("sessionId")
    if not session_id:
        return jsonify({"error": "Missing sessionId"}), 400

    questions = data.get("summary", {}).get("questions", [])
    if not questions:
        return jsonify({"error": "Missing summary.questions schema"}), 400

    # Flatten and normalize the questionnaire schema
    all_fields = flatten_all_fields(questions)
    for field in all_fields:
        if "value" not in field:
            field["value"] = ""

    # Save to Session Store (Valkey/Redis)
    state_payload = {
        "sessionId": session_id,
        "source": data.get("source", "unknown"),
        "all_fields": all_fields
    }
    set_session_state(session_id, state_payload)

    log.info(f"Session {session_id} initialized with {len(all_fields)} fields.")
    return jsonify({"status": "success", "message": f"Session {session_id} initialized."}), 200


@app.route("/session/chunk", methods=["POST"])
def session_chunk():
    """
    Receives a new conversation chunk and drops it into AWS SQS FIFO.
    Expects payload: { "sessionId": "...", "source": "...", "conversation": [ ... ] }
    """
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    session_id = data.get("sessionId")
    conversation = data.get("conversation", [])

    if not session_id:
        return jsonify({"error": "Missing sessionId"}), 400
    if not conversation:
        return jsonify({"error": "Missing conversation chunk"}), 400

    # Verify session exists in our store
    state = get_session_state(session_id)
    if not state:
        return jsonify({"error": f"Session {session_id} not initialized. Call /session/init first."}), 404

    # Push to Redis queue
    # The redis_worker daemon will pick this up and process it asynchronously.
    queue_payload = {
        "sessionId": session_id,
        "source": data.get("source", state.get("source", "unknown")),
        "chunk_turns": conversation
    }
    
    msg_id = push_to_redis_queue(queue_payload)
    
    log.info(f"Chunk received for {session_id}. Enqueued to Redis with ID: {msg_id}")
    
    # Return 202 Accepted immediately so the frontend isn't blocked by LLM processing
    return jsonify({
        "status": "accepted",
        "message": "Chunk queued for processing",
        "message_id": msg_id
    }), 202


if __name__ == "__main__":
    # Run development server
    log.info("Starting ArcaQuest Ingestion API on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=True)
