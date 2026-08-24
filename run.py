"""
Entry point for the ArcaQuest Stream Processor.

Run from any location:
    python run.py
    python run.py --input payloads/nutrition_attitude.json
    python run.py --input /absolute/path/to/custom.json

Default input: payloads/food_intake.json
Output:        outputs/output_result.json
               outputs/chunks_log.json
"""

import argparse
import os
import sys

# Ensure the project directory is on the path so imports work from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stream_processor import run_workflow

_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT = os.path.join(_DIR, "payloads", "food_intake.json")

AVAILABLE_PAYLOADS = [
    "food_intake",
    "exercise_patterns",
    "physical_activity_knowledge",
    "nutrition_attitude",
    "nutrition_knowledge",
    "dietary_recall",
    "physical_activity_attitude",
    "food_frequency_questionnaire",
]


def main():
    parser = argparse.ArgumentParser(
        description="ArcaQuest Stream Processor — chunk-based transcript reranker workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available built-in payloads (in payloads/):\n"
            + "\n".join(f"  {p}" for p in AVAILABLE_PAYLOADS)
        ),
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Path to the input JSON file (conversation + questionnaire schema). "
            "Can be a filename in payloads/ or an absolute path. "
            f"Default: payloads/food_intake.json"
        ),
    )
    args = parser.parse_args()

    # Resolve input path: check as-is, then relative to payloads/
    input_path = args.input
    if not os.path.isabs(input_path):
        # Try relative to CWD first
        if os.path.exists(input_path):
            input_path = os.path.abspath(input_path)
        else:
            # Try as a name inside payloads/
            candidate = os.path.join(_DIR, "payloads", input_path)
            if not candidate.endswith(".json"):
                candidate += ".json"
            if os.path.exists(candidate):
                input_path = candidate
            else:
                input_path = os.path.abspath(input_path)
    else:
        input_path = os.path.abspath(input_path)

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        print(f"        Available payloads: {', '.join(AVAILABLE_PAYLOADS)}")
        sys.exit(1)

    print(f"Input  : {input_path}")
    print(f"Output : {os.path.join(_DIR, 'outputs', 'output_result.json')}")
    print()
    run_workflow(input_path)


if __name__ == "__main__":
    main()
