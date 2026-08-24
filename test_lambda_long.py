import requests

API_URL = "https://4xhbovakz4.execute-api.ap-south-1.amazonaws.com/invoke"

payload = {
    "prompt": "Doctor: I’d like to ask you a few questions to check your knowledge about nutrition.\nPatient: Sure, let’s start.\nDoctor: Whole plant foods are foods as they exist in nature. Is that correct?\nPatient: Yes.\nDoctor: Good. Now, choose a food with high carbohydrate content.\nPatient: Rice.\n\nExtract the answer for: Whole plant foods are foods as they exist in nature",
    "sessionId": "test_isolation_002"
}

print(f"Pinging API Gateway at {API_URL} with longer prompt...")
try:
    response = requests.post(API_URL, json=payload, timeout=60)
    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    print(response.json())
except Exception as e:
    print(f"Error connecting to API Gateway: {e}")
