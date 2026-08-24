# ArcaQuest AWS Stream Processor
### Technical Documentation — ArcaQuest POC

---

## 1. Problem Statement

During a long clinical consultation (typically 30–40 minutes), a clinician relies on an AI system to automatically fill a medical questionnaire from the conversation. Two key failure modes exist:

| Failure Mode | Impact |
|---|---|
| **Missed questions** — LLM could not extract an answer, or the clinician forgot to ask | Clinician must re-read the full transcript or re-ask the patient |
| **Patient self-correction** — Patient gives an answer early, then corrects it later | Old (incorrect) answer stays in the system; final record is wrong |

Both scenarios lead to inaccurate patient data and unnecessary clinician burden.

---

## 2. Solution Overview

A **streaming, chunk-based processing pipeline** that:

1. Processes the conversation in **sliding-window chunks** as it arrives (4 speaker turns, 1-turn overlap)
2. Uses a **reranker model** to intelligently select which questionnaire fields are most relevant to each chunk — avoiding unnecessary LLM calls
3. Answers selected fields with **direct LLM calls** (Amazon Nova Pro) using **type-aware prompts**
4. **Detects patient self-corrections** — if a patient revises an earlier answer, the stored value is automatically updated
5. Handles **all Lifestyle Questionnaire field types**: radio, conditional, slider, frequency, textarea, number, select, conditional_select, section, time_range
6. **Flags complex types** (table, grouped_table, searchable_select, dependent_autofill) that require UI/API interaction — skipped cleanly and reported in the log
7. Produces a fully filled `outputs/output_result.json` and a detailed per-chunk audit log

---

## 3. Pipeline Architecture

```
Transcript (conversation JSON)
         │
         ▼
┌─────────────────────────────────────┐
│   Sliding-Window Chunker            │
│   Size: 4 turns │ Overlap: 1 turn   │
└─────────────────────────────────────┘
         │
         ▼  (for each chunk)
┌─────────────────────────────────────────────────────────────┐
│   Reranker Call                                             │
│                                                             │
│   Query  = raw chunk text (speaker turns concatenated)      │
│   Docs   = ALL answerable fields (answered + unanswered)    │
│              formatted by field_handlers.field_to_reranker_doc() │
│                                                             │
│   Output = ranked field list with relevance scores          │
└─────────────────────────────────────────────────────────────┘
         │
         ├──── Unanswered fields (top-K by score) ────────────►  Step A
         │
         └──── Answered fields (score ≥ 0.55 threshold) ──────►  Step B
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│   Step A: Answer Extraction (Unanswered Fields)             │
│                                                             │
│   For each top-ranked unanswered field:                     │
│     → Type-aware prompt built by field_handlers             │
│     → Direct Nova Pro call with chunk as context            │
│     → 3-attempt retry with exponential backoff (429)        │
│     → Type-aware answer cleaning by field_handlers          │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│   Step B: Self-Correction Detection (Answered Fields)        │
│                                                             │
│   For each answered field with score ≥ 0.55:               │
│     → Re-evaluate via Nova Pro using this chunk             │
│     → If new answer ≠ stored answer → CORRECTION applied    │
│     → If same answer or no info → keep existing value       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────┐
│   Progressive Chunk Log Update    │
│   outputs/chunks_log.json         │
└───────────────────────────────────┘
         │
         ▼  (after all chunks)
┌───────────────────────────────────┐
│   outputs/output_result.json      │
│   Final filled questionnaire      │
└───────────────────────────────────┘
```

---

## 4. Key Design Decisions

### Why no per-chunk summarization?

An earlier design considered summarizing each chunk with the LLM before sending to the reranker. This was **rejected** because:
- It adds 1 extra LLM call per chunk → increased latency and cost
- Summaries compress and lose specific detail → reranker scores become less accurate
- Raw chunk text (≈80 words per chunk) is small enough for direct LLM prompting

### Why unanswered and answered fields are separated after reranking

All fields (answered + unanswered) are sent to the reranker in one call. However, the ranked results are split **before** selecting top-K:

- **Unanswered fields** get their own dedicated top-K slots for answering
- **Answered fields** are evaluated separately for corrections only if score ≥ threshold

This prevents already-answered fields from crowding out unanswered fields in the top-K selection — an issue that caused 3 out of 9 fields to be missed in early testing.

### Why direct LLM calls instead of FAISS / RAG per chunk?

Each chunk is only 4 speaker turns (~80 words). Building a FAISS index on such a small context and running retrieval is unnecessary overhead. A single, direct LLM call with the chunk as context is faster and more accurate for this size.

### Why a separate `field_handlers.py` module?

The Lifestyle Payloads corpus uses 10+ distinct field types. Each type requires:
- A different reranker document format (how the question is described to the reranker)
- A different LLM prompt structure
- A different answer-extraction strategy (regex, option matching, phrase lookup, etc.)

Separating this logic into `field_handlers.py` keeps `stream_processor.py` clean and makes adding new field types trivial — only one file needs to change.

### Why complex types are skipped

`table`, `grouped_table`, `searchable_select`, `dependent_autofill` are UI-driven or API-driven:
- `table` / `grouped_table`: Require structured row-by-row entry, not single-value extraction
- `searchable_select`: Backed by a live nutrition database API
- `dependent_autofill`: Value auto-populated from another field's API lookup

These types are flagged with `[SKIP-COMPLEX]` in logs and listed in the final summary. The clinician fills them manually in the UI after the session.

---

## 5. Technical Specifications

### Models & APIs

| Component | Details |
|---|---|
| **Reranker Model** | `danielchalef/Qwen3-Reranker-4B-seq-cls-vllm-fixed` |
| **Reranker Endpoint** | `https://94u5v92s73.execute-api.ap-south-1.amazonaws.com/dev/v1/rerank` |
| **LLM** | Amazon Nova Pro (`us.amazon.nova-pro-v1:0`) |
| **LLM Region** | `us-east-1` via AWS Bedrock Converse API |
| **Auth** | `BEDROCK_API_KEY` (Bearer token, loaded from `.env`) |

### Chunking Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Chunk size | 4 speaker turns | Captures a full Q&A exchange with context |
| Overlap | 1 turn | Ensures boundary answers are not missed |
| Step size | 3 turns | `chunk_size - overlap` |

### Answer Extraction

| Parameter | Value |
|---|---|
| Top-K fields per chunk | 4 (unanswered) |
| Max LLM attempts per field | 3 |
| Correction threshold (reranker score) | 0.55 |
| Rate limit backoff | 2s → 4s → 8s (exponential, on HTTP 429) |

---

## 6. Supported Field Types

### Fully Supported (Extracted by LLM)

| Type | Prompt Strategy | Answer Extraction |
|---|---|---|
| `radio` | One option from fixed list | Regex whole-word match on options |
| `conditional` | Yes/No with optional subfield | "yes"/"no" detection; subfield number extraction |
| `slider` | Integer within min–max range | First integer in valid range |
| `frequency` | Pick from Daily/Weekly/Monthly/Rarely/Never grid | Canonical value via phrase-to-code map |
| `textarea` | Free-text open-ended | First paragraph, max 200 chars |
| `number` | Numeric value (optional min/max) | First numeric match |
| `select` | Single-choice dropdown | Same as radio — regex match on options |
| `conditional_select` | Dropdown depending on another field | Options dict flattened; same as radio |
| `section` | Container — sub-fields extracted recursively | Sub-fields processed individually |
| `field_group` | Flat container — sub-fields extracted | Sub-fields processed individually |
| `time_range` | HH:MM–HH:MM window | Regex time pattern |

### Complex Types (Skipped — Manual Review Required)

| Type | Reason |
|---|---|
| `table` | Multi-row exercise activity grid; requires row-by-row UI entry |
| `grouped_table` | Dietary recall with live food database linkage |
| `searchable_select` | Backed by nutrition API (`/nutrition/v0.1/food/list`) |
| `dependent_autofill` | Auto-populated from serving-unit API |
| `icon` | Display-only; no answer needed |

---

## 7. Answer Cleaning (field_handlers.py)

LLMs occasionally append reasoning or explanation after the answer. `clean_answer()` strips this:

| Field Type | Raw LLM Output | Stored Value |
|---|---|---|
| `radio` | `"3\n\nExplanation: The patient said..."` | `"3"` |
| `conditional` | `"Yes\n\nReasoning: Patient mentioned..."` | `"Yes"` |
| `slider` | `"I'd say 7 out of 10"` | `"7"` |
| `frequency` | `"The patient eats this three times a week"` | `"weekly_3"` |
| `textarea` | `"The patient stated they want to reduce sugar..."` | `"The patient stated they want to reduce sugar..."` (trimmed to 200 chars) |
| `number` | `"About 2 cups"` | `"2"` |

---

## 8. Section Field Handling

`section` and `field_group` containers are **flattened** at load time by `flatten_all_fields()` in `field_handlers.py`. Each leaf sub-field is treated as an independent field in the reranker pool.

Example — `nutrition_attitude.json`:
```
section: "reason_for_choice"
  └── radio: "knowledge_of_healthy_food"       ← Becomes independent field
  └── radio: "availability_of_healthy_food"    ← Becomes independent field
  └── radio: "affordability_of_healthy_food"   ← Becomes independent field
  ...

section: "nutritional_changes_ranking"
  └── textarea: "Nutritional Change 1"          ← Becomes independent field
  └── slider:   "How important is it?"         ← Becomes independent field
  └── slider:   "How confident are you?"       ← Becomes independent field
  ...
```

---

## 9. Edge Cases Handled

### Patient Self-Correction
**Scenario:** Patient says "I skip breakfast" in chunk 2, then corrects to "Actually I skip lunch" in chunk 6.

**How it's handled:**
- Chunk 6's reranker scores the `missed_meal` field above the correction threshold (0.55)
- The LLM is re-called with chunk 6 as context
- New answer `"Lunch"` differs from stored `"Breakfast"` → **correction applied**
- Logged in `outputs/chunks_log.json` under `"corrections"` with old and new values

### Unanswered Fields at End
- `outputs/output_result.json` — fields with `value: ""` are clearly visible
- `outputs/chunks_log.json` — full per-chunk audit
- Console summary — explicit list of unanswered field IDs and labels with types

### API Rate Limiting (HTTP 429)
```
Attempt 1 → 429 → wait 2s → Attempt 2 → 429 → wait 4s → Attempt 3 → fail gracefully
```

---

## 10. Output Files

### `outputs/output_result.json`
The original questionnaire JSON with all `value` fields populated. Unanswered fields have `value: ""`.

### `outputs/chunks_log.json`
Per-chunk audit log. Overwritten fresh at run start; appended progressively.

**Structure per chunk entry:**
```json
{
  "chunk_number": 3,
  "turns": [{ "speaker": "doctor", "message": "..." }, ...],
  "chunk_text": "Doctor: ...\nPatient: ...",
  "unanswered_count": 6,
  "answered_count": 3,
  "reranker_top_unanswered": [
    { "rank": 1, "score": 0.6260, "field_id": "major_meals_per_day",
      "field_type": "radio", "label": "Number of major meals per day" }
  ],
  "reranker_correction_candidates": [
    { "score": 0.5800, "field_id": "tea", "field_type": "conditional",
      "label": "Tea", "current_value": "Yes" }
  ],
  "answers_found": [
    { "rank": 1, "field_id": "major_meals_per_day", "field_type": "radio",
      "label": "Number of major meals per day", "answer": "3" }
  ],
  "corrections": [
    { "field_id": "missed_meal", "field_type": "radio",
      "label": "Which meal do you skip?", "old_answer": "Breakfast", "new_answer": "Lunch" }
  ]
}
```

---

## 11. Project File Structure

```
ArcaQuest_AWS_Stream_Processing/
├── .env                              ← BEDROCK_API_KEY
├── run.py                            ← CLI entry point
├── stream_processor.py               ← Core pipeline logic
├── field_handlers.py                 ← Type-specific field formatting & extraction
├── README.md                         ← Quick-start guide
├── workflow_documentation.md         ← This document
├── architecture_diagram.png          ← System architecture diagram
├── reranker_llm_flow.png             ← Reranker + LLM flow diagram
├── payloads/                         ← All 8 test payloads
│   ├── food_intake.json              ← radio, conditional, slider
│   ├── exercise_patterns.json        ← table (complex)
│   ├── physical_activity_knowledge.json  ← radio (15 fields)
│   ├── nutrition_attitude.json       ← section, radio, slider, textarea
│   ├── nutrition_knowledge.json      ← radio (21 fields)
│   ├── dietary_recall.json           ← grouped_table, field_group (complex)
│   ├── physical_activity_attitude.json   ← section, radio, slider, textarea
│   └── food_frequency_questionnaire.json ← frequency (68 fields)
└── outputs/                          ← Auto-generated on first run
    ├── output_result.json
    └── chunks_log.json
```

### Run Commands

```bash
cd ArcaQuest_AWS_Stream_Processing

# Default (food_intake.json — same as original chunk_reranker test):
python run.py

# Any payload by name:
python run.py --input nutrition_attitude
python run.py --input food_frequency_questionnaire

# Full path:
python run.py --input payloads/physical_activity_knowledge.json
python run.py --input /absolute/path/to/custom.json
```

---

## 12. Test Results (Original chunk_reranker baseline)

Tested against the `food_intake.json` payload (19-turn dietary consultation, 9 fields):

| Metric | Result |
|---|---|
| Total turns | 17 |
| Chunks generated | 6 |
| Fields answered | **9 / 9** |
| Patient self-corrections detected | Correctly tracked |
| Rate limit errors (429) | Handled gracefully with backoff |
| Unanswered fields | 0 |
| Run time (approx.) | ~90 seconds |

---

## 13. Future Enhancements (Deferred)

| Enhancement | Description |
|---|---|
| **Global fallback pass** | After all chunks, run full-transcript FAISS retrieval for remaining unanswered fields |
| **Streaming mode** | Trigger chunk processing in real-time as conversation is recorded |
| **Multi-questionnaire routing** | Route chunks to the right section of a larger, multi-section questionnaire |
| **Confidence scoring** | Attach confidence score to each answer (reranker score + LLM certainty) |
| **Complex type extraction** | Attempt structured extraction for table/grouped_table via few-shot prompting |

---

*Document prepared by: ArcaQuest Engineering Team*
*Last updated: August 2026*
*Codebase: ArcaQuest_AWS_Stream_Processing (replaces test/chunk_reranker)*
