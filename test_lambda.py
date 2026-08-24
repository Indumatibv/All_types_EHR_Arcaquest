import requests

API_URL = "https://4xhbovakz4.execute-api.ap-south-1.amazonaws.com/invoke"

payload = {
    "prompt": "Hello! Are you the ArcaQuest medical assistant?",
    "sessionId": "test_isolation_001"
}

print(f"Pinging API Gateway at {API_URL}...")
try:
    response = requests.post(API_URL, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    print(response.json())
except Exception as e:
    print(f"Error connecting to API Gateway: {e}")
