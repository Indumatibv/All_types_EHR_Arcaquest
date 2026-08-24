# ArcaQuest AWS Stream Processor

A chunk-based transcript reranker pipeline that extracts answers for **all Lifestyle Questionnaire field types** from doctor-patient conversation recordings.

Supports: `radio`, `conditional`, `slider`, `frequency`, `textarea`, `number`, `select`, `conditional_select`, `section`, `time_range` — and flags `table`, `grouped_table`, `searchable_select`, `dependent_autofill` as requiring manual review.

---

## Quick Start

### 1. Install dependencies

```bash
pip install requests python-dotenv
```

### 2. Verify `.env`

The `.env` file already contains your `BEDROCK_API_KEY`. No changes needed for local testing.

### 3. Run

```bash
# From this directory:
cd ArcaQuest_AWS_Stream_Processing

python run.py                                                    # Default: food_intake
python run.py --input payloads/food_intake.json                  # Explicit
python run.py --input payloads/nutrition_attitude.json           # Section + slider + textarea
python run.py --input payloads/physical_activity_knowledge.json  # Radio-only, 15 fields
python run.py --input payloads/food_frequency_questionnaire.json # Frequency type
python run.py --input payloads/exercise_patterns.json            # Table type (complex)
python run.py --input payloads/nutrition_knowledge.json          # Radio-only, 21 fields
python run.py --input payloads/dietary_recall.json               # Grouped table (complex)
python run.py --input payloads/physical_activity_attitude.json   # Section + radio + slider

# Shorthand (payload name without path):
python run.py --input nutrition_attitude
python run.py --input food_frequency_questionnaire
```

---

## Available Payloads

All payloads are extracted from the **Lifestyle Payloads docx** and live in `payloads/`:

| File | Field Types | Turns |
|---|---|---|
| `food_intake.json` | radio, conditional, slider | 17 |
| `exercise_patterns.json` | table (complex) | 19 |
| `physical_activity_knowledge.json` | radio (15 fields) | 33 |
| `nutrition_attitude.json` | radio, section, slider, textarea | 31 |
| `nutrition_knowledge.json` | radio (21 fields) | 45 |
| `dietary_recall.json` | grouped_table, field_group (complex) | 8 |
| `physical_activity_attitude.json` | radio, section, slider, textarea | 8 |
| `food_frequency_questionnaire.json` | frequency (68 fields) | 21 |

---

## Output Files

Both are written to `outputs/`:

| File | Description |
|---|---|
| `outputs/output_result.json` | Original JSON with all `value` fields populated |
| `outputs/chunks_log.json` | Per-chunk audit log (overwritten each run) |

---

## Project Structure

```
ArcaQuest_AWS_Stream_Processing/
├── .env                              ← BEDROCK_API_KEY
├── run.py                            ← CLI entry point
├── stream_processor.py               ← Core pipeline (chunker, reranker, LLM loop)
├── field_handlers.py                 ← Type-specific field formatting & answer extraction
├── workflow_documentation.md         ← Full technical documentation
├── architecture_diagram.png          ← System architecture diagram
├── reranker_llm_flow.png             ← Reranker + LLM flow diagram
├── payloads/                         ← All 8 test payloads from Lifestyle Payloads docx
│   ├── food_intake.json
│   ├── exercise_patterns.json
│   ├── physical_activity_knowledge.json
│   ├── nutrition_attitude.json
│   ├── nutrition_knowledge.json
│   ├── dietary_recall.json
│   ├── physical_activity_attitude.json
│   └── food_frequency_questionnaire.json
└── outputs/                          ← Auto-generated (created on first run)
    ├── output_result.json
    └── chunks_log.json
```

---

## Config

All tuneable constants are at the top of `stream_processor.py`:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 4 | Speaker turns per chunk |
| `CHUNK_OVERLAP` | 1 | Overlapping turns between chunks |
| `TOP_K` | 4 | Top-N fields selected by reranker per chunk |
| `CORRECTION_THRESHOLD` | 0.55 | Reranker score threshold for re-evaluating answered fields |
| `max_attempts` | 3 | LLM retry attempts per field |

---

## Complex Field Types

Some UI-heavy field types cannot be reliably extracted from a plain conversation transcript:

| Type | Reason | Handling |
|---|---|---|
| `table` | Multi-row exercise activity grid | Logged with `[SKIP-COMPLEX]`, skipped |
| `grouped_table` | Dietary recall with API-linked food items | Logged with `[SKIP-COMPLEX]`, skipped |
| `searchable_select` | Requires live food database search | Logged with `[SKIP-COMPLEX]`, skipped |
| `dependent_autofill` | Auto-populated from API response | Logged with `[SKIP-COMPLEX]`, skipped |

These fields are clearly listed in the console output and can be filled manually in the output JSON.
