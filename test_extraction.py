import json
from stream_processor import try_answer_field

def test():
    with open("payloads/nutrition_knowledge.json", "r") as f:
        knowledge = json.load(f)
        
    fields = knowledge["summary"]["questions"][0]["fields"]
    
    # Find nuts field
    nuts_field = next(f for f in fields if f.get("id") == "nuts_expensive")
    
    chunk = """
    Patient: Steaming.
    Doctor: Right. Eating nuts is expensive with marginal benefits—do you agree?
    Patient: No.
    Doctor: Good answer. Are artificial sweeteners safe and healthy?
    """
    
    from stream_processor import try_answer_field, build_answer_prompt
    
    prompt = build_answer_prompt(chunk, nuts_field)
    print("PROMPT:")
    print(prompt)
    print("-" * 50)
    ans = try_answer_field(chunk, nuts_field, "test_session")
    print("Nuts field Answer:", ans)

test()
