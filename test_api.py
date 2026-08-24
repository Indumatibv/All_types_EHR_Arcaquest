import requests
import json
import time

URL_INIT = "http://localhost:5001/session/init"
URL_CHUNK = "http://localhost:5001/session/chunk"

# Read the full schema from food_intake.json
with open("payloads/food_intake.json", "r") as f:
    data = json.load(f)

session_id = "test_sess_123"

# 1. INIT
init_payload = {
    "sessionId": session_id,
    "source": "nutrition",
    "summary": data["summary"]
}
print("Initializing session...")
r1 = requests.post(URL_INIT, json=init_payload)
print(r1.status_code, r1.json())
time.sleep(1)

# 2. CHUNK
chunk_payload = {
    "sessionId": session_id,
    "source": "nutrition",
    "conversation": data["conversation"][:3] # Just the first 3 turns
}
print("Sending chunk 1...")
r2 = requests.post(URL_CHUNK, json=chunk_payload)
print(r2.status_code, r2.json())
