import requests
import json
import time

URL_INIT = "http://localhost:5001/session/init"
URL_CHUNK = "http://localhost:5001/session/chunk"

# Read the full schema from nutrition_knowledge.json
with open("payloads/nutrition_knowledge.json", "r") as f:
    data = json.load(f)

session_id = "nutrition_test_999"

# 1. INIT
init_payload = {
    "sessionId": session_id,
    "source": "nutrition_knowledge",
    "summary": data["summary"]
}
print(f"Initializing session '{session_id}'...")
r1 = requests.post(URL_INIT, json=init_payload)
print(r1.status_code, r1.json())
time.sleep(1)

# 2. CHUNK STREAMING (Simulate frontend sending chunks as the patient talks)
conversation = data["conversation"]
chunk_size = 4
overlap = 1

print(f"Sending full conversation ({len(conversation)} turns) in chunks...")

# Sliding window chunking to simulate the streaming frontend
step = chunk_size - overlap
for i in range(0, len(conversation), step):
    chunk_turns = conversation[i : i + chunk_size]
    if not chunk_turns:
        break
        
    chunk_payload = {
        "sessionId": session_id,
        "source": "nutrition_knowledge",
        "conversation": chunk_turns
    }
    
    print(f"Sending chunk ({len(chunk_turns)} turns)...")
    r2 = requests.post(URL_CHUNK, json=chunk_payload)
    print(r2.status_code, r2.json())
    
    # Pause slightly to simulate real-time talking
    time.sleep(2)
