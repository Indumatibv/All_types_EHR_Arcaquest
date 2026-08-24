import json
import boto3
import time

AGENT_RUNTIME_ARN = "arn:aws:bedrock-agentcore:ap-south-1:829876691474:runtime/ahs_ehr_staging_medical_assistant-VjzRMU8wzX"

client = boto3.client("bedrock-agentcore", region_name="ap-south-1")

def lambda_handler(event, context):
    """
    AWS Lambda handler for invoking the Bedrock Agent Runtime.
    Expected JSON payload in event body:
    {
        "prompt": "The prompt text to send to the agent...",
        "sessionId": "The unique session ID (will be padded to 33 chars if shorter)"
    }
    """
    start_time = time.time()

    try:
        # Parse the incoming event body
        body = json.loads(event.get("body") or "{}")
        prompt = body.get("prompt")
        session_id = body.get("sessionId", "default_session_id")

        if not prompt:
            return _response(400, {"error": "'prompt' is required"})

        # Bedrock Agent Runtime requires the sessionId to be at least 33 chars long.
        runtime_session_id = session_id
        if len(runtime_session_id) < 33:
            runtime_session_id = runtime_session_id.ljust(33, '0')

        print(f"Sending payload to Bedrock Agent (Session: {runtime_session_id})")
        print(f"Prompt content:\n{prompt}\n{'-'*40}")

        # Invoke the Agent
        response = client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeSessionId=runtime_session_id,
            payload=json.dumps({"prompt": prompt}).encode("utf-8"),
            contentType="application/json",
            accept="application/json"
        )

        stream = response.get("response")
        if not stream:
            return _response(500, {"error": "No response body found from Bedrock Agent."})

        # Parse the Agent's response stream
        result_text = stream.read().decode("utf-8")
        result_json = json.loads(result_text)

        # Sometimes Bedrock returns a JSON string inside a JSON string
        if isinstance(result_json, str):
            result_json = json.loads(result_json)

        # Extract the actual text response based on the Agent's schema
        agent_answer = ""
        if isinstance(result_json, dict):
            if "response" in result_json:
                agent_answer = str(result_json["response"]).strip()
            elif "text" in result_json:
                agent_answer = str(result_json["text"]).strip()
            elif "generation" in result_json:
                agent_answer = str(result_json["generation"]).strip()
            else:
                agent_answer = str(result_json).strip()
        else:
            agent_answer = str(result_json).strip()

        latency_ms = (time.time() - start_time) * 1000

        return _response(200, {
            "answer": agent_answer,
            "latency_ms": latency_ms
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _response(500, {"error": str(e)})


def _response(status_code, body_dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"  # Tighten this in production!
        },
        "body": json.dumps(body_dict)
    }
