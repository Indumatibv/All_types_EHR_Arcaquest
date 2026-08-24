# ArcaQuest Streaming API Architecture Walkthrough

I have successfully refactored the codebase to follow the production **AWS SQS FIFO + Valkey** architecture diagram. 

The pipeline is now decoupled into two scalable, highly available microservices that will prevent HTTP timeouts and safely process chunks in order.

## What I Built

### 1. `api.py` (The API Server)
A lightweight Flask REST API that your frontend team will hit.
- **`POST /session/init`**: Saves the empty questionnaire schema for the session.
- **`POST /session/chunk`**: Instead of waiting 15 seconds for the LLM, this endpoint drops the payload into the AWS SQS queue and returns `202 Accepted` instantly.

### 2. `sqs_worker.py` (The Async Processor)
A background daemon meant to run continuously on your EKS pods.
- It pulls messages from SQS.
- Runs the Reranker + Nova LLM pipeline using the new `process_single_chunk` function.
- Saves the state back to Valkey.
- Constructs the **exact Delta JSON response** you requested, containing only the specific fields that were updated.

### 3. `aws_mock.py` (Local Development Mocks)
Since you do not have AWS SQS and Valkey running on your local machine, I wrote a transparent mocking layer that saves the queues and states into a `.local_state/` folder. This means you can run the entire pipeline right now on your laptop without any AWS configuration!

## How to Test Locally

You need three terminal windows to see the real-time interaction.

**Terminal 1 (Start the API)**
```bash
cd /Users/admin/ai-questionnaire-project/Arcaquest_POC/AWS/ArcaQuest_AWS_Stream_Processing
python api.py
```

**Terminal 2 (Start the Worker Daemon)**
```bash
cd /Users/admin/ai-questionnaire-project/Arcaquest_POC/AWS/ArcaQuest_AWS_Stream_Processing
python sqs_worker.py
```

**Terminal 3 (Simulate the Frontend)**
I have created a test script that acts like the frontend. Run it to send the `init` and `chunk` commands.
```bash
python test_api.py
```

Watch **Terminal 2** to see the worker pick up the chunk from the queue, run the LLM, and log out the exact JSON payload format it will push back to the frontend!

> [!NOTE]
> To push this to production, your DevOps team will just need to swap out the functions in `aws_mock.py` with standard `boto3.client('sqs')` and `redis.Redis()` calls. The architecture logic itself is complete!
