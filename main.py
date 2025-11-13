from asyncio import tasks
import json
import os
import logging
import uuid
import asyncio
import tracemalloc
import psutil
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from openai import AzureOpenAI
import tempfile
import re
import time

# -----------------------
# Config
# -----------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
# -----------------------
async def generate_summary_async(transcript_path: str, output_file: str):
    try:
        client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint="https://arcaquest-emr.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )

        # --- load transcript text ---
        with open(transcript_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()
        if not full_text:
            logging.warning("Transcript empty. No summary generated.")
            return ""

        # --- create prompt ---
        prompt = f"""
        Summary:
        You are given a conversation between an interviewer and a participant.
        The interviewer asks various questions about the participant's personal life, habits, diet, health, and daily routine.
        Write a detailed summary **only** from the participant's point of view using "I" statements.
        Transcript:
        {full_text}
        """

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        # --- save and return summary ---
        summary = response.choices[0].message.content.strip()
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(summary)

        logging.info(f"✅ Summary saved to {output_file}")
        return summary

    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return ""

# -----------------------
# Build retriever using FAISS per request
# -----------------------
def build_retriever(summary_file: str):
    try:
        logging.info(f"Building FAISS vectorstore from {summary_file}")
        loader = TextLoader(summary_file, encoding="utf-8")
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)
        embedding = AzureOpenAIEmbeddings(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint="https://arcaquest-emr.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        if not docs:
            logging.warning("No documents to build FAISS retriever. Skipping retriever.")
            return None
        vectorstore = FAISS.from_documents(docs, embedding)

        logging.info("✅ FAISS Retriever ready")
        return vectorstore
    except Exception as e:
        logging.error(f"Error building FAISS retriever: {e}")
        raise

# -----------------------
# Async process single frequency field
# -----------------------

async def process_single_frequency_field(dictionary, qa_chain, request_id):
    try:
        ehr_question = dictionary["label"]
        options = [
            "daily",
            "weekly_1", "weekly_2", "weekly_3", "weekly_4", "weekly_5",
            "monthly_1", "monthly_2",
            "rarely",
            "never"
        ]
        options_str = ",".join(options)

        explanation = (
            "Use the following meanings for frequency codes: "
            "weekly_1 = weekly once, weekly_2 = weekly twice, weekly_3 = weekly three times, "
            "weekly_4 = weekly four times, weekly_5 = weekly five times, "
            "monthly_1 = monthly once, monthly_2 = monthly twice. "
            "Other codes are self-explanatory (daily, rarely, never)."
        )

        question = (
            f"{explanation} "
            f"Select the most appropriate answer from the given options only. "
            f"USE ONLY THE PROVIDED CONTEXT TO ANSWER. Do NOT generate new options or free text. "
            f"If the answer is not found, respond with 'No information found'. "
            f"If multiple options are mentioned in the context, select the **first matching option**. "
            f"Question: {ehr_question}, Options: {options_str}"
        )

        dictionary["value"] = "No information found"
        max_attempts = 3
        total_calls = 0

        def is_no_info(text: str) -> bool:
            """Returns True if LLM output means 'No information found'."""
            text_clean = re.sub(r"[^a-zA-Z ]", "", text).strip().lower()
            return "no information found" in text_clean or "no info found" in text_clean

        # 🔁 Retry loop — ensure 3 LLM calls for "No information found"
        for attempt in range(max_attempts):
            total_calls += 1
            result = await asyncio.to_thread(qa_chain.invoke, question)
            answer = (result.get("result", "") or "").strip()

            if not answer or is_no_info(answer):
                logging.warning(
                    f"[{request_id}] Attempt {attempt+1}: No info for frequency '{ehr_question}', retrying..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.5)
                    continue
                else:
                    answer = "No information found"
                    logging.warning(
                        f"[{request_id}] All {max_attempts} attempts failed for '{ehr_question}'. Keeping 'No information found'."
                    )
            else:
                logging.info(
                    f"[{request_id}] ✅ LLM attempt {attempt+1} succeeded for '{ehr_question}' → {answer}"
                )
                dictionary["value"] = answer
                break

        # 🧾 Summary log
        logging.info(
            f"[{request_id}] 🧾 Frequency summary → {ehr_question}: "
            f"Value='{dictionary['value']}' after {total_calls} LLM calls "
            f"({attempt+1 if not is_no_info(dictionary['value']) else 'failed all attempts'})"
        )

        # ✅ Final completion log
        logging.info(
            f"[{request_id}] ✅ Updated frequency field '{ehr_question}' with value '{dictionary['value']}' "
            f"(Total LLM calls: {total_calls})"
        )

    except Exception as e:
        logging.error(f"[{request_id}] Error processing frequency field '{dictionary.get('label', '')}': {e}")


# -----------------------
# Async process single radio field
# -----------------------

async def process_single_radio_field(dictionary, qa_chain, request_id):
    try:
        ehr_question = dictionary["label"]
        options = ",".join(dictionary["options"])
        question = (
            f"Select the most appropriate answer from the given options only. "
            f"USE ONLY THE PROVIDED CONTEXT TO ANSWER. Do NOT generate new options or free text. "
            f"If the answer is not found, respond with 'No information found'. "
            f"If multiple options are mentioned in the context, select the **first matching option** "
            f"from the list below. Question: {ehr_question}, Options: {options}"
        )

        dictionary["value"] = "No information found"
        max_attempts = 3
        llm_calls = 0

        def is_no_info(text: str) -> bool:
            """Helper: Detect 'No information found' variations."""
            clean_text = re.sub(r"[^a-zA-Z ]", "", text).strip().lower()
            return "no information found" in clean_text or "no info found" in clean_text

        # 🔁 Always attempt 3 LLM calls if "No information found"
        for attempt in range(max_attempts):
            llm_calls += 1
            result = await asyncio.to_thread(qa_chain.invoke, question)
            answer = (result.get("result", "") or "").strip()

            if not answer or is_no_info(answer):
                logging.warning(
                    f"[{request_id}] Attempt {attempt+1}: No info for radio '{ehr_question}', retrying..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.5)
                    continue
                else:
                    answer = "No information found"
                    logging.warning(
                        f"[{request_id}] All {max_attempts} attempts failed for radio '{ehr_question}'. Keeping 'No information found'."
                    )
            else:
                logging.info(
                    f"[{request_id}] ✅ LLM attempt {attempt+1} succeeded for radio '{ehr_question}' → {answer}"
                )
                dictionary["value"] = answer
                break

        # 🧾 Summary log
        logging.info(
            f"[{request_id}] 🧾 Radio summary → {ehr_question}: "
            f"Value='{dictionary['value']}' after {llm_calls} LLM calls"
        )

        # ✅ Final log
        logging.info(
            f"[{request_id}] ✅ Completed radio field '{ehr_question}' "
            f"with value '{dictionary['value']}' (Total LLM calls: {llm_calls})"
        )

    except Exception as e:
        logging.error(f"[{request_id}] Error processing radio field '{dictionary.get('label', '')}': {e}")


# -----------------------
# Async process single conditional field
# -----------------------

async def process_single_conditional_field(dictionary, qa_chain, request_id):
    try:
        ehr_question = dictionary["label"]

        # Extract options
        condition_labels = [c["label"] for c in dictionary.get("conditions", [])]
        options = ",".join(condition_labels)

        # Build main question
        question_main = (
            f"Select the most appropriate answer from the given options only. "
            f"USE ONLY THE PROVIDED CONTEXT TO ANSWER. Do NOT generate new options or free text. "
            f"If the answer is not found, respond with 'No information found'. "
            f"If multiple options are mentioned, pick the first one. "
            f"Question: {ehr_question}, Options: {options}"
        )

        dictionary["value"] = "No information found"
        max_attempts = 3
        llm_calls = 0

        def is_no_info(text: str) -> bool:
            """Returns True if LLM output means 'No information found' in any form."""
            text_clean = re.sub(r"[^a-zA-Z ]", "", text).strip().lower()
            return "no information found" in text_clean or "no info found" in text_clean

        # 🔁 Retry up to 3 times if "No information found"
        for attempt in range(max_attempts):
            llm_calls += 1
            result = await asyncio.to_thread(qa_chain.invoke, question_main)
            selected_value = (result.get("result", "") or "").strip()

            if not selected_value or is_no_info(selected_value):
                logging.warning(
                    f"[{request_id}] Attempt {attempt+1}: No info for conditional '{ehr_question}', retrying..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.5)
                    continue  # try again
                else:
                    selected_value = "No information found"
                    logging.warning(
                        f"[{request_id}] All {max_attempts} attempts failed for '{ehr_question}'. Keeping 'No information found'."
                    )
            else:
                logging.info(
                    f"[{request_id}] ✅ LLM attempt {attempt+1} succeeded for '{ehr_question}' → {selected_value}"
                )
                dictionary["value"] = selected_value
                break

        # Handle sub-field (if Yes)
        yes_condition = next(
            (c for c in dictionary["conditions"] if c["label"].lower() == "yes"), None
        )

        if yes_condition and dictionary["value"].lower() == "yes" and "subField" in yes_condition:
            sub_field = yes_condition["subField"]
            sub_question = (
                f"The participant mentioned '{ehr_question} = Yes'. "
                f"Now, determine: {sub_field['label']}. "
                f"Return only a number between {sub_field.get('min', 0)} and {sub_field.get('max', 10)} "
                f"if mentioned. Otherwise, return 'No information found'."
            )

            sub_field["value"] = "No information found"
            for attempt in range(max_attempts):
                llm_calls += 1
                sub_result = await asyncio.to_thread(qa_chain.invoke, sub_question)
                sub_answer = (sub_result.get("result", "") or "").strip()

                if not sub_answer or is_no_info(sub_answer):
                    logging.warning(
                        f"[{request_id}] Attempt {attempt+1}: No info for subfield '{sub_field['label']}', retrying..."
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        sub_answer = "No information found"
                        logging.warning(
                            f"[{request_id}] All {max_attempts} attempts failed for subfield '{sub_field['label']}'."
                        )
                else:
                    sub_field["value"] = sub_answer
                    logging.info(
                        f"[{request_id}] ✅ LLM attempt {attempt+1} succeeded for subfield '{sub_field['label']}' → {sub_answer}"
                    )
                    break

            logging.info(f"[{request_id}] → SubField '{sub_field['label']}' = {sub_field['value']}")

        # 🧾 Conditional summary log
        logging.info(
            f"[{request_id}] 🧾 Conditional summary → {ehr_question}: "
            f"Value='{dictionary['value']}' after {llm_calls} LLM calls "
            f"({attempt+1 if not is_no_info(dictionary['value']) else 'failed all attempts'})"
        )

        # ✅ Final completion log
        logging.info(
            f"[{request_id}] ✅ Completed conditional field '{ehr_question}' "
            f"(Total LLM calls: {llm_calls})"
        )

    except Exception as e:
        logging.error(
            f"[{request_id}] Error processing conditional field '{dictionary.get('label', '')}': {e}"
        )

# -----------------------
# Async process single table field
# -----------------------

async def process_single_table_field(dictionary, qa_chain, request_id):
    try:
        table_label = dictionary.get("label", "")
        headers = dictionary.get("headers", [])
        value_rows = []

        llm_calls = 0  # 👈 Track total LLM calls
        max_attempts = 3

        def is_no_info(text: str) -> bool:
            """Helper: Detect variations of 'No information found'."""
            clean_text = re.sub(r"[^a-zA-Z ]", "", text).strip().lower()
            return "no information found" in clean_text or "no info found" in clean_text

        # Step 1️⃣ — Ask LLM to identify exercise activities
        base_question = (
            f"From the conversation, identify all physical or exercise activities the participant mentions. "
            f"For each activity, extract details like type (Aerobics/Strength/Flexibility/Balance), "
            f"activity name, duration (in minutes if mentioned), intensity (Mild/Moderate/Intense), "
            f"frequency (Daily, 3/4/5/6 days a week), and days if specific days are mentioned. "
            f"Respond as a list of structured entries. "
            f"If any information is missing, fill 'No information found'."
        )

        parsed_text = "No information found"
        for attempt in range(max_attempts):
            llm_calls += 1
            result = await asyncio.to_thread(qa_chain.invoke, base_question)
            parsed_text = (result.get("result", "") or "").strip()

            if not is_no_info(parsed_text):
                logging.info(
                    f"[{request_id}] ✅ LLM attempt {attempt+1} succeeded for '{table_label}' base extraction."
                )
                break

            logging.warning(
                f"[{request_id}] Attempt {attempt+1}: No info found for '{table_label}', retrying base extraction..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(0.5)
            else:
                logging.warning(
                    f"[{request_id}] All {max_attempts} attempts failed for base extraction of '{table_label}'."
                )

        logging.info(f"[{request_id}] 🧾 Base extracted text for '{table_label}':\n{parsed_text}")

        # Step 2️⃣ — Reformat into JSON
        reform_question = (
            f"Convert the following exercise details into a JSON array where each item contains keys: "
            f"{[h['id'] for h in headers]}. Ensure matching option values from the given headers. "
            f"If not sure, use 'No information found'. Keep days as list if multiple mentioned.\n"
            f"Text:\n{parsed_text}"
        )

        reform_result_text = "No information found"
        for attempt in range(max_attempts):
            llm_calls += 1
            reform = await asyncio.to_thread(qa_chain.invoke, reform_question)
            reform_result_text = (reform.get('result', '') or '').strip()

            if not is_no_info(reform_result_text):
                logging.info(
                    f"[{request_id}] ✅ LLM attempt {attempt+1} succeeded for '{table_label}' JSON reformat."
                )
                break

            logging.warning(
                f"[{request_id}] Attempt {attempt+1}: No info found for '{table_label}', retrying JSON reformat..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(0.5)
            else:
                logging.warning(
                    f"[{request_id}] All {max_attempts} attempts failed for JSON reformat of '{table_label}'."
                )

        # Step 3️⃣ — Parse JSON or fallback
        try:
            value_rows = json.loads(reform_result_text)
            if not isinstance(value_rows, list):
                raise ValueError("Parsed result is not a list")
        except Exception as e:
            logging.error(f"[{request_id}] JSON parsing failed for '{table_label}': {e}")
            value_rows = [{"activity_type": "No information found"}]

        dictionary["value"] = value_rows

        # ✅ Final summary log
        logging.info(
            f"[{request_id}] 🧾 Table summary → {table_label}: "
            f"Rows={len(value_rows)} | Total LLM calls={llm_calls}"
        )

        logging.info(
            f"[{request_id}] ✅ Completed table field '{table_label}' "
            f"with {len(value_rows)} rows (Total LLM calls: {llm_calls})"
        )

    except Exception as e:
        logging.error(
            f"[{request_id}] ❌ Error processing table field '{dictionary.get('label', '')}': {e}"
        )

# -----------------------
# Async process single grouped table field (SMART TIME-AWARE VERSION)
# -----------------------

async def process_single_grouped_table_field(dictionary, qa_chain, request_id):
    try:
        table_label = dictionary.get("label", "")
        meal_groups = dictionary.get("mealGroups", [])
        meal_labels = [m["label"] for m in meal_groups]

        base_question = (
            f"You are a structured data extraction model.\n\n"
            f"From the conversation, extract foods the person consumed during the day.\n"
            f"Group them under these exact meal names: {meal_labels}.\n\n"
            f"For each meal, extract every mentioned food item with the following details:\n"
            f"- time_range: {{'from': <start_time>, 'to': <end_time>}}\n"
            f"    * Acceptable formats: '8 am', '08:30', '10 pm', etc.\n"
            f"    * If only start time mentioned (e.g., 'around 8'), set that as 'from' and leave 'to' empty.\n"
            f"    * If only end time mentioned (e.g., 'till 9'), set that as 'to' and leave 'from' empty.\n"
            f"    * If AM/PM is missing, infer from context words like 'morning', 'noon', 'evening', 'night'.\n"
            f"- food_item: the name of the food consumed.\n"
            f"- serving_type: unit like 'plate', 'bowl', 'cup', etc.\n"
            f"- quantity: numeric value if mentioned.\n"
            f"- icon: keep empty string.\n\n"
            f"If a meal isn't mentioned, include it with an empty rows array.\n\n"
            f"Output must be **valid JSON array** exactly like this:\n"
            f"[\n"
            f"  {{'id': 'breakfast', 'label': 'Breakfast', 'rows': [\n"
            f"      {{'time_range': {{'from': '8 am', 'to': '9 am'}}, 'food_item': 'Oats porridge', 'serving_type': 'bowl', 'quantity': 1, 'icon': ''}},\n"
            f"      {{'time_range': {{'from': '9:15 am', 'to': ''}}, 'food_item': 'Banana', 'serving_type': 'piece', 'quantity': 1, 'icon': ''}}\n"
            f"  ]}},\n"
            f"  {{'id': 'lunch', 'label': 'Lunch', 'rows': []}},\n"
            f"  ...\n"
            f"]\n\n"
            f"Return only valid JSON — no markdown or commentary."
        )

        result = await asyncio.to_thread(qa_chain.invoke, base_question)
        raw = result["result"].strip()

        try:
            start = raw.find("[")
            end = raw.rfind("]")
            if start == -1 or end == -1:
                raise ValueError("No JSON array detected")
            parsed = json.loads(raw[start:end+1])
        except Exception as e:
            logging.warning(f"[{request_id}] Could not parse grouped table JSON: {e}")
            parsed = []

        # ensure structure for every meal group
        meal_map = {m["label"]: {"id": m["id"], "label": m["label"], "rows": []} for m in meal_groups}
        for group in parsed:
            label = group.get("label")
            if label in meal_map:
                for row in group.get("rows", []):
                    time_range = row.get("time_range", {"from": "", "to": ""})

                    # --- normalize time strings ---
                    def normalize_time(t):
                        if not t:
                            return ""
                        t = t.strip().lower().replace(".", "")
                        # Add am/pm inference if missing
                        if not any(x in t for x in ["am", "pm"]):
                            if any(k in row["food_item"].lower() for k in ["breakfast", "morning"]):
                                t += " am"
                            elif any(k in row["food_item"].lower() for k in ["evening", "dinner", "night"]):
                                t += " pm"
                        return t

                    time_range["from"] = normalize_time(time_range.get("from", ""))
                    time_range["to"] = normalize_time(time_range.get("to", ""))
                    row["time_range"] = time_range

                meal_map[label]["rows"] = group.get("rows", [])

        dictionary["value"] = list(meal_map.values())
        logging.info(f"[{request_id}] ✅ Populated grouped table '{table_label}' with {sum(len(g['rows']) for g in meal_map.values())} rows")

    except Exception as e:
        logging.error(f"[{request_id}] Error processing grouped table '{dictionary.get('label', '')}': {e}")

# -----------------------

UNIT_CONVERSIONS = {
    "mg->g": 0.001,
    "g->mg": 1000,
    "kg->g": 1000,
    "g->kg": 0.001,
    "ml->l": 0.001,
    "l->ml": 1000
}

def extract_expected_unit(label: str):
    m = re.search(r"\(([^)]+)\)", label)
    if not m:
        return None
    text = m.group(1).lower()
    if "mg" in text: return "mg"
    if "gm" in text or "g/" in text: return "g"
    if "kg" in text: return "kg"
    if "ml" in text: return "ml"
    if "l" in text: return "l"
    return None

def convert_unit(value, from_unit, to_unit):
    if from_unit == to_unit or value in ("", "No information found"):
        return value
    key = f"{from_unit}->{to_unit}"
    if key in UNIT_CONVERSIONS:
        return round(float(value) * UNIT_CONVERSIONS[key], 3)
    return value

# -----------------------
# Async process single filed group
# -----------------------

async def process_single_field_group(dictionary, qa_chain, request_id):
    import json, time, asyncio, logging

    # === Extract meta info ===
    group_label = dictionary.get("id", "")
    fields = dictionary.get("fields", [])
    field_ids = [f["id"] for f in fields]
    field_labels = [f.get("label", f["id"]) for f in fields]

    # expected units extracted dynamically
    expected_units = {f["id"]: extract_expected_unit(f.get("label", "")) for f in fields}

    # === Prompt ===
    base_prompt = (
        f"You are an information extraction model.\n\n"
        f"Extract numeric daily intake values mentioned in the conversation for each of these fields:\n"
        f"{json.dumps(field_labels, indent=2)}.\n\n"
        f"For each field, you must identify:\n"
        f"- The numeric value (e.g. 1, 30, 40)\n"
        f"- The unit mentioned in the conversation (e.g. g, mg, ml)\n"
        f"- If no value is mentioned, respond with 'No information found'.\n\n"
        f"The expected units are:\n"
        f"{json.dumps(expected_units, indent=2)}.\n\n"
        f"If the unit differs, convert it accordingly (mg ↔ g, l ↔ ml, etc.).\n\n"
        f"Return output strictly as valid JSON mapping field ids to objects with 'value' and 'unit' keys.\n"
        f"If a field is not mentioned, still include it with 'value': 'No information found' and 'unit': null.\n"
        f"Return only valid JSON — no extra text."
    )

    # initialize structure
    extracted = {fid: {"value": "No information found", "unit": None} for fid in field_ids}
    llm_calls = 0
    max_attempts = 3

    # === Retry loop ===
    for attempt in range(max_attempts):
        try:
            llm_calls += 1
            result = await asyncio.to_thread(qa_chain.invoke, base_prompt)
            raw = result["result"].strip()
            logging.info(f"[{request_id}] LLM call {llm_calls} for group '{group_label}' → Raw output: {raw[:300]}...")

            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON detected")
            raw_json = raw[start:end+1]
            extracted = json.loads(raw_json)

            # ✅ Stop retrying if at least one field got a valid value
            if not all(v.get("value") == "No information found" for v in extracted.values()):
                break

            logging.warning(f"[{request_id}] All values 'No information found' (attempt {attempt+1})")

        except Exception as e:
            logging.warning(f"[{request_id}] Parse fail (attempt {attempt+1}): {e}")
            await asyncio.sleep(0.5)

    # === Postprocess & conversions ===
    final_values = {}
    for fid in field_ids:
        val_info = extracted.get(fid, {})
        val = val_info.get("value", "No information found")
        from_unit = (val_info.get("unit") or "").lower()
        to_unit = expected_units.get(fid)

        # numeric conversion if applicable
        if isinstance(val, (int, float, str)) and str(val).replace(".", "", 1).isdigit():
            try:
                val = convert_unit(float(val), from_unit, to_unit)
            except Exception:
                pass
        if isinstance(val, float) and val.is_integer():
            val = int(val)

        final_values[fid] = val

    dictionary["value"] = final_values

    # === Final logs ===
    logging.info(f"[{request_id}] ✅ Final '{group_label}' -> {final_values}")
    logging.info(f"[{request_id}] 💬 Total LLM calls for '{group_label}': {llm_calls}")

# -----------------------
# Async process single slider field
# -----------------------
async def process_single_slider_field(dictionary, qa_chain, request_id, conversation_text=None, conversation=None, related_text=None):
    import re, logging, asyncio

    label = dictionary.get("label", "")
    field_id = dictionary.get("id", "")

    # 🧠 Build context
    conversation_context = ""
    if conversation:
        for entry in conversation:
            speaker = entry.get("speaker", "").capitalize()
            message = entry.get("message", "")
            conversation_context += f"{speaker}: {message}\n"
    elif conversation_text:
        conversation_context = conversation_text

    extra_context = f"\nThis question refers to the change: '{related_text}'" if related_text else ""

    # 📝 Prompt
    prompt = f"""
You are analyzing a doctor-patient conversation.
Given the conversation below, extract the **numeric value (0–10)** that corresponds to this question:

**Question:** "{label}"{extra_context}

If the patient gave an approximate or descriptive answer (e.g., "around 7", "maybe 8"),
return just the numeric value.
If the question wasn't answered, return "No information found".

Conversation:
{conversation_context}

Respond ONLY with the number (integer), or the phrase "No information found".
"""

    # 🧩 Defaults
    value = "No information found"
    total_llm_calls = 0
    max_attempts = 3

    # 🔁 Retry loop
    for attempt in range(max_attempts):
        total_llm_calls += 1
        try:
            result = await qa_chain.ainvoke({"query": prompt})
            raw_output = result.get("result", "").strip() if isinstance(result, dict) else str(result).strip()
            logging.debug(f"[{request_id}] Raw LLM output for '{label}' (attempt {attempt+1}): {raw_output}")

            # 🧮 Extract number safely
            match = re.search(r"\b([0-9]|10)\b", raw_output)
            if match:
                value = int(match.group(1))
            elif "no information" in raw_output.lower():
                value = "No information found"
            else:
                value = "No information found"

            # ✅ Stop retrying if numeric found
            if isinstance(value, int):
                logging.info(f"[{request_id}] ✅ LLM attempt {attempt+1} succeeded for '{label}' → {value}")
                break

            if attempt < max_attempts - 1:
                logging.warning(f"[{request_id}] Attempt {attempt+1}: No info found for '{label}', retrying...")
                await asyncio.sleep(0.5)

        except Exception as e:
            logging.error(f"[{request_id}] ❌ Error during slider extraction (attempt {attempt+1}): {e}")
            await asyncio.sleep(0.5)

    # 🧾 Final summary
    logging.info(f"[{request_id}] ✅ Final slider '{label}' → {value} (Total LLM calls: {total_llm_calls})")

    dictionary["value"] = value
    return dictionary

# -----------------------
# Async process single textarea field
# -----------------------

async def process_single_textarea_field(dictionary, qa_chain, request_id, conversation_text, conversation, change_context):
    """
    Extracts textarea values like 'Nutritional Change 1' and stores them in change_context
    so that related sliders (importance/confidence) can reference the correct change.
    Always makes exactly 3 LLM calls if responses are 'No information found' or empty.
    If final value is 'No information found', linked sliders should also be 'No information found'.
    """
    import logging, asyncio

    label = dictionary.get("label", "")
    field_id = dictionary.get("id", "")

    # 🧠 Build conversation context
    conversation_context = ""
    if conversation:
        for entry in conversation:
            speaker = entry.get("speaker", "").capitalize()
            message = entry.get("message", "")
            conversation_context += f"{speaker}: {message}\n"
    elif conversation_text:
        conversation_context = conversation_text

    # 🧩 Build LLM prompt
    prompt = f"""
You are analyzing a doctor-patient conversation.
Extract the patient's mentioned **nutritional change** corresponding to this question:

**Question:** "{label}"

Conversation:
{conversation_context}

Respond with the exact change (short phrase like "Eat healthy" or "Cut down sugar").
If no change was mentioned, respond with "No information found".
"""

    value = "No information found"
    total_llm_calls = 0
    max_attempts = 3

    # 🔁 Retry loop
    for attempt in range(1, max_attempts + 1):
        total_llm_calls += 1
        try:
            response = await qa_chain.ainvoke({"query": prompt})
            raw_output = (
                response.get("result", "").strip()
                if isinstance(response, dict)
                else str(response).strip()
            )
            logging.debug(f"[{request_id}] Raw LLM output (attempt {attempt}) for '{label}': {raw_output}")

            # 🚫 Treat "No information found" as failure — trigger retry
            if not raw_output or "no information found" in raw_output.lower() or raw_output.lower() in ["none", "n/a"]:
                logging.warning(f"[{request_id}] Attempt {attempt}: No info for '{label}', retrying..." if attempt < max_attempts else f"[{request_id}] All {max_attempts} attempts failed for '{label}'.")
                await asyncio.sleep(0.4)
                continue  # ❌ try again

            # ✅ Got valid result
            value = raw_output
            logging.info(f"[{request_id}] ✅ LLM attempt {attempt} succeeded for '{label}' → {value}")
            break

        except Exception as e:
            logging.error(f"[{request_id}] ❌ Error during textarea extraction (attempt {attempt}): {e}")
            await asyncio.sleep(0.4)

    # ✅ Final summary
    logging.info(f"[{request_id}] ✅ Final textarea '{label}' value: {value} (Total LLM calls: {total_llm_calls})")

    dictionary["value"] = value

    # 🔗 Save in context for related slider fields
    if "nutritional_change_" in field_id.lower():
        change_context.setdefault("changes", []).append(value)
        if value.lower() == "no information found":
            change_context.setdefault("invalid_changes", []).append(field_id)

    return dictionary

# -----------------------

# Run tasks in batches to limit concurrency
async def run_tasks_in_batches(tasks, batch_size=30):
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    return results

# -----------------------
# Async process fields with parallel radios and frequency
# -----------------------

async def process_fields_async(dict_list, qa_chain, request_id, conversation_text=None, conversation=None):
    tasks = []
    change_context = {"changes": []}  # stores textarea context for related sliders

    # STEP 1: Process non-slider, especially textareas first (sequentially)
    for dictionary in dict_list:
        field_type = dictionary.get("type")

        if field_type == "textarea":
            await process_single_textarea_field(
                dictionary, qa_chain, request_id, conversation_text, conversation, change_context
            )

        elif "fields" in dictionary and isinstance(dictionary["fields"], list):
            await process_fields_async(
                dictionary["fields"], qa_chain, request_id, conversation_text, conversation
            )

    # STEP 2: Process sliders (importance/confidence) after textareas are done
    for dictionary in dict_list:
        field_type = dictionary.get("type")

        if field_type == "slider" and dictionary["id"].startswith(("importance_change_", "confidence_change_")):
            idx = int(dictionary["id"].split("_")[-1]) - 1
            related_text = change_context["changes"][idx] if idx < len(change_context.get("changes", [])) else None
            tasks.append(
                process_single_slider_field(
                    dictionary, qa_chain, request_id, conversation_text, conversation, related_text
                )
            )

        elif field_type == "slider":
            tasks.append(
                process_single_slider_field(
                    dictionary, qa_chain, request_id, conversation_text, conversation
                )
            )

        elif field_type == "radio" and "options" in dictionary:
            tasks.append(process_single_radio_field(dictionary, qa_chain, request_id))

        elif field_type == "frequency":
            tasks.append(process_single_frequency_field(dictionary, qa_chain, request_id))

        elif field_type == "conditional" and "conditions" in dictionary:
            tasks.append(process_single_conditional_field(dictionary, qa_chain, request_id))

        elif field_type == "table":
            tasks.append(process_single_table_field(dictionary, qa_chain, request_id))

        elif field_type == "grouped_table":
            tasks.append(process_single_grouped_table_field(dictionary, qa_chain, request_id))

        elif field_type == "field_group":
            tasks.append(process_single_field_group(dictionary, qa_chain, request_id))

    # STEP 3: Run sliders & other parallelizable fields
    if tasks:
        await run_tasks_in_batches(tasks, batch_size=30)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint for Kubernetes probes.
    Returns 200 OK if the app is running.
    """
    return {"status": "ok"}

@app.post("/process-ehr-json/")
async def process_ehr_json(request: Request):
    async with semaphore:  # Limit concurrent requests
        return await asyncio.wait_for(_process_logic(request), timeout=300)

# -----------------------
# Main processing logic
# -----------------------
async def _process_logic(request: Request):
    request_id = uuid.uuid4().hex
    logging.info(f"[{request_id}] Received request")
    tracemalloc.start()
    process = psutil.Process(os.getpid())

    transcript_file = summary_file = None

    try:
        try:
            updated_json = await request.json()
        except Exception:
            logging.error(f"[{request_id}] Invalid JSON body")
            raise HTTPException(status_code=400, detail="Invalid JSON body.")

        conversation = updated_json.get("conversation", [])
        if not conversation:
            logging.warning(f"[{request_id}] No conversation found in JSON")
            raise HTTPException(status_code=400, detail="No conversation data found in JSON.")

        with tempfile.NamedTemporaryFile(delete=False) as transcript_f:
            transcript_file = transcript_f.name
        with tempfile.NamedTemporaryFile(delete=False) as summary_f:
            summary_file = summary_f.name

        # Write transcript
        with open(transcript_file, "w", encoding="utf-8") as f:
            for entry in conversation:
                speaker = entry.get("speaker", "").capitalize()
                message = entry.get("message", "")
                f.write(f"{speaker}: {message}\n")

        # Generate summary
        await generate_summary_async(transcript_file, summary_file)


        # Build FAISS retriever
        vectorstore = build_retriever(summary_file)
        if not vectorstore:  # Check if retriever was successfully built
            logging.warning(f"[{request_id}] FAISS retriever not built, skipping QA chain.")
            return updated_json

        retriever = vectorstore.as_retriever()

        llm = AzureChatOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint="https://arcaquest-emr.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_deployment="o4-mini",
            # temperature=0.2,     # <-- Set temperature here
        )

        template = """
        answer ONLY from the provided context.
        Context:
        {context}
        Question:
        {question}
        Answer:
        - If the answer is clearly in the context, return it.
        - If it is not, return exactly: "No information found."
        """
        prompt = PromptTemplate.from_template(template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

        questions = updated_json.get("summary", {}).get("questions", [])
        logging.info(f"[{request_id}] Processing {len(questions)} questions")

        question_tasks = []

        for content in questions:
            if "fields" in content and "sections" not in content:
                # question_tasks.append(process_fields_async(content["fields"], qa_chain, request_id))
                question_tasks.append(process_fields_async(content["fields"], qa_chain, request_id, conversation_text="\n".join([f"{c['speaker']}: {c['message']}" for c in conversation]), conversation=conversation))

            if "sections" in content:
                for fields_list in content["sections"]:
                    if "fields" in fields_list:
                        question_tasks.append(process_fields_async(fields_list["fields"], qa_chain, request_id))

        # if question_tasks:
        #     await asyncio.gather(*question_tasks)
        if question_tasks:
            await run_tasks_in_batches(question_tasks, batch_size=5)

    finally:
        # Cleanup FAISS and temp files
        if 'vectorstore' in locals():
            del vectorstore
        if transcript_file and os.path.exists(transcript_file):
            os.remove(transcript_file)
        if summary_file and os.path.exists(summary_file):
            os.remove(summary_file)

        mem_mb = process.memory_info().rss / 1024**2
        logging.info(f"[{request_id}][Memory] Process RSS after request: {mem_mb:.2f} MB")

        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"[{request_id}][tracemalloc] Final memory - Current: {current / 1024**2:.2f} MB, Peak: {peak / 1024**2:.2f} MB")
        tracemalloc.stop()

    logging.info(f"[{request_id}] ✅ Finished processing JSON")
    return updated_json
# -----------------------