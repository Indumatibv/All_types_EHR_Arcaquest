"""
field_handlers.py
=================

Type-specific helpers for every questionnaire field type found in the
Lifestyle Payloads corpus.

Supported types
---------------
  radio              — fixed-option single-choice
  conditional        — Yes/No with optional subField (number)
  slider             — integer on a numeric scale
  frequency          — frequency grid (Daily / Weekly-N / Monthly-N / Rarely / Never)
  textarea           — open free-text answer
  number             — plain numeric answer
  select             — single-choice dropdown (treated like radio)
  section            — container; sub-fields are extracted via flatten_section_fields()
  conditional_select — dropdown that depends on another field value (treated like radio)
  time_range         — time window (e.g. "8:00–10:00")
  grouped_table      — dietary recall rows (complex; flagged for manual review)
  table              — exercise activity table (complex; flagged for manual review)
  searchable_select  — food search dropdown (complex; flagged for manual review)
  dependent_autofill — auto-filled from API (complex; flagged for manual review)
  field_group        — flat group of sub-fields (recursively processed)
  icon               — display-only; always skipped

Complex types (table, grouped_table, searchable_select, dependent_autofill)
are detected by is_complex_type() and skipped by the reranker loop.
"""

import re
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Field types that are UI-heavy or API-driven and cannot be reliably extracted
# from a plain conversation transcript.
COMPLEX_TYPES = {"table", "grouped_table", "searchable_select", "dependent_autofill",
                 "time_range", "icon"}

# Frequency column vocabulary → canonical value stored in 'value'
FREQUENCY_COLUMN_MAP = {
    "daily":     "daily",
    "weekly_1":  "weekly_1",
    "weekly_2":  "weekly_2",
    "weekly_3":  "weekly_3",
    "weekly_4":  "weekly_4",
    "weekly_5":  "weekly_5",
    "monthly_1": "monthly_1",
    "monthly_2": "monthly_2",
    "rarely":    "rarely",
    "never":     "never",
}

# Human-readable phrases that map to canonical frequency values
FREQUENCY_PHRASE_MAP = [
    (r"\bevery\s+day\b|\bdaily\b",          "daily"),
    (r"\b5\s+times?\s+a?\s*week\b",         "weekly_5"),
    (r"\b4\s+times?\s+a?\s*week\b",         "weekly_4"),
    (r"\b3\s+times?\s+a?\s*week\b",         "weekly_3"),
    (r"\bthrice\b",                          "weekly_3"),
    (r"\b2\s+times?\s+a?\s*week\b|\btwice\s+a\s+week\b",  "weekly_2"),
    (r"\bonce\s+a\s+week\b|\b1\s+time\s+a\s+week\b",      "weekly_1"),
    (r"\btwice\s+a\s+month\b|\b2\s+times?\s+a?\s*month\b","monthly_2"),
    (r"\bonce\s+a\s+month\b|\b1\s+time\s+a\s+month\b",    "monthly_1"),
    (r"\brarely\b|\boccasionally\b|\bsometimes\b",         "rarely"),
    (r"\bnever\b|\bnot\s+at\s+all\b",                     "never"),
]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Field-type helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_complex_type(field: dict) -> bool:
    """Return True for types that require UI interaction or API calls."""
    return field.get("type") in COMPLEX_TYPES


def flatten_section_fields(field: dict) -> List[dict]:
    """
    For a 'section' or 'field_group' container, return a flat list of its
    leaf sub-fields (skip further nested containers — handled recursively
    by the caller).
    """
    sub_fields = []
    for sf in field.get("fields", []):
        ft = sf.get("type", "")
        if ft in ("section", "field_group"):
            sub_fields.extend(flatten_section_fields(sf))
        elif ft not in COMPLEX_TYPES:
            sub_fields.append(sf)
    return sub_fields


def flatten_all_fields(questions: List[dict]) -> List[dict]:
    """
    Return a flat list of every answerable leaf field across all question
    groups.  Handles:
      - Normal groups: questions[i].fields[j]
      - Section / field_group containers: recursively flattened
      - Section-based groups: questions[i].sections[k].fields[j]
      - Complex types: excluded
    """
    fields = []
    seen_ids = set()
    
    def _add_field(field_list):
        for field in field_list:
            ft = field.get("type", "")
            if ft in ("section", "field_group"):
                _add_field(flatten_section_fields(field))
            elif ft not in COMPLEX_TYPES:
                fid = field.get("id")
                if fid and fid not in seen_ids:
                    seen_ids.add(fid)
                    fields.append(field)
                elif not fid:
                    fields.append(field)

    for group in questions:
        # Top-level fields list
        _add_field(group.get("fields", []))

        # Section-based groups (e.g. food_frequency_questionnaire)
        for section in group.get("sections", []):
            _add_field(section.get("fields", []))
            
    return fields


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Reranker document formatter
# ──────────────────────────────────────────────────────────────────────────────

def field_to_reranker_doc(field: dict) -> str:
    """
    Format a questionnaire field as a descriptive string for the reranker.
    Returns a compact, information-dense representation of what the field
    is asking for.
    """
    label      = field.get("label", "")
    field_type = field.get("type", "")

    if field_type == "radio":
        opts = ", ".join(str(o) for o in field.get("options", []))
        return f"{label} (options: {opts})"

    elif field_type == "conditional":
        conditions  = field.get("conditions", [])
        cond_labels = ", ".join(c["label"] for c in conditions)
        sub_info    = ""
        for c in conditions:
            if "subField" in c:
                sf       = c["subField"]
                sub_info = (f" — if {c['label']}: {sf['label']} "
                            f"({sf.get('min', 0)}–{sf.get('max', 10)})")
        return f"{label} ({cond_labels}){sub_info}"

    elif field_type == "slider":
        min_v = field.get("min", 0)
        max_v = field.get("max", 10)
        desc  = field.get("description", "")
        return f"{label} (scale {min_v}–{max_v} {desc})".strip()

    elif field_type in ("select", "conditional_select"):
        options = field.get("options", [])
        if isinstance(options, list):
            opts = ", ".join(str(o) for o in options)
        elif isinstance(options, dict):
            # conditional_select: flatten all sub-option lists
            all_opts = []
            for sub_list in options.values():
                all_opts.extend(str(o) for o in sub_list)
            opts = ", ".join(all_opts)
        else:
            opts = str(options)
        return f"{label} (select: {opts})"

    elif field_type == "frequency":
        columns = field.get("columns", [])
        headers = ", ".join(c.get("header", "") for c in columns)
        return f"{label} (frequency: {headers})"

    elif field_type == "textarea":
        placeholder = field.get("placeholder", "")
        if placeholder:
            return f"{label} (free text: {placeholder})"
        return f"{label} (free text response)"

    elif field_type == "number":
        min_v = field.get("min", "")
        max_v = field.get("max", "")
        range_str = f" ({min_v}–{max_v})" if min_v != "" and max_v != "" else ""
        return f"{label} (number{range_str})"

    elif field_type == "time_range":
        return f"{label} (time range, e.g. 8:00–10:00)"

    # Fallback
    return label


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

def build_answer_prompt(chunk_text: str, field: dict) -> str:
    """
    Build a tightly-scoped LLM prompt for a single field.
    Returns a string prompt ready to send to Nova Pro.
    """
    label      = field.get("label", "")
    field_type = field.get("type", "")

    header = (
        "You are analyzing a specific excerpt of a doctor-patient conversation.\n"
        "CRITICAL RULE: You MUST answer using ONLY the text in the 'Conversation Excerpt' below.\n"
        "Do NOT use your memory, prior conversation history, or any knowledge outside this excerpt.\n"
        "CRITICAL RULE 2: You MUST ONLY extract an answer if the PATIENT explicitly provides it in this exact excerpt. If the doctor mentions a topic but the patient has not yet answered it, or if the patient is answering a different question, you must respond with exactly: No information found\n"
        "CRITICAL RULE 3: If the doctor makes a statement and asks 'do you agree?' or 'is that correct?' and the patient responds 'Yes' or 'No', that 'Yes' or 'No' IS the patient's explicit answer to the topic.\n\n"
        f"Conversation Excerpt:\n{chunk_text}\n\n"
    )

    if field_type == "radio":
        opts = ", ".join(str(o) for o in field.get("options", []))
        return (
            header
            + f"Question: {label}\n"
            + f"Options: {opts}\n\n"
            + "Select ONLY one option exactly as written above.\n"
            + "If the answer is not clearly in the context, respond with exactly:\n"
            + "No information found\n\nAnswer:"
        )

    elif field_type == "conditional":
        conditions = field.get("conditions", [])
        opts       = ", ".join(c["label"] for c in conditions)
        return (
            header
            + f"Question: {label}\n"
            + f"Options: {opts}\n\n"
            + "Select ONLY one option from the list above.\n"
            + "If the answer is not clearly in the context, respond with exactly:\n"
            + "No information found\n\nAnswer:"
        )

    elif field_type == "slider":
        min_v = field.get("min", 0)
        max_v = field.get("max", 10)
        desc  = field.get("description", "")
        return (
            header
            + f"Question: {label} {desc}\n"
            + f"Return only a number between {min_v} and {max_v}.\n"
            + "If the answer is not clearly in the context, respond with exactly:\n"
            + "No information found\n\nAnswer:"
        )

    elif field_type in ("select", "conditional_select"):
        options = field.get("options", [])
        if isinstance(options, list):
            opts = ", ".join(str(o) for o in options)
        elif isinstance(options, dict):
            # Present all sub-options flattened
            all_opts = []
            for sub_list in options.values():
                all_opts.extend(str(o) for o in sub_list)
            opts = ", ".join(all_opts)
        else:
            opts = str(options)
        return (
            header
            + f"Question: {label}\n"
            + f"Options: {opts}\n\n"
            + "Select ONLY one option exactly as written above.\n"
            + "If the answer is not clearly in the context, respond with exactly:\n"
            + "No information found\n\nAnswer:"
        )

    elif field_type == "frequency":
        columns = field.get("columns", [])
        # Build human-readable column descriptions
        col_desc = []
        for c in columns:
            header_text = c.get("header", "")
            opts = c.get("option", [])
            col_desc.append(f"{header_text} ({', '.join(opts)})")
        cols_str = " | ".join(col_desc)
        return (
            header
            + f"Question: How often does the patient consume: {label}?\n"
            + f"Frequency options: {cols_str}\n\n"
            + "Return ONLY the canonical option value (e.g. 'daily', 'weekly_3', 'rarely', 'never').\n"
            + "If the answer is not clearly in the context, respond with exactly:\n"
            + "No information found\n\nAnswer:"
        )

    elif field_type == "textarea":
        placeholder = field.get("placeholder", "")
        hint = f" ({placeholder})" if placeholder else ""
        return (
            header
            + f"Question: {label}{hint}\n"
            + "Provide a concise, direct answer based ONLY on what the patient said.\n"
            + "If the answer is not clearly in the context, respond with exactly:\n"
            + "No information found\n\nAnswer:"
        )

    elif field_type == "number":
        min_v = field.get("min", "")
        max_v = field.get("max", "")
        range_str = f" Return a number between {min_v} and {max_v}." if min_v != "" and max_v != "" else " Return only a number."
        return (
            header
            + f"Question: {label}\n"
            + range_str + "\n"
            + "If the answer is not clearly in the context, respond with exactly:\n"
            + "No information found\n\nAnswer:"
        )

    elif field_type == "time_range":
        return (
            header
            + f"Question: {label}\n"
            + "Return a time range in HH:MM–HH:MM format (24-hour).\n"
            + "If the answer is not clearly in the context, respond with exactly:\n"
            + "No information found\n\nAnswer:"
        )

    # Generic fallback
    return (
        header
        + f"Question: {label}\n"
        + "If the answer is not clearly in the context, respond with exactly:\n"
        + "No information found\n\nAnswer:"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Answer cleaner
# ──────────────────────────────────────────────────────────────────────────────

def is_no_info(text: str) -> bool:
    """Detect all variants of 'No information found'."""
    clean = re.sub(r"[^a-zA-Z ]", "", text).strip().lower()
    return (
        "no information found" in clean
        or "no info found" in clean
        or clean in ("none", "na", "n a")
    )


def clean_answer(raw: str, field: dict) -> str:
    """
    Extract just the core answer value from potentially verbose LLM output.

    Type-specific extraction:
      radio / select / conditional_select  → first matching option (case-insensitive)
      conditional                          → 'Yes' or 'No' from first line
      slider / number                      → first integer found
      frequency                            → canonical frequency value
      textarea                             → first non-empty paragraph (up to 200 chars)
      time_range                           → HH:MM–HH:MM pattern
      fallback                             → first non-empty line
    """
    if not raw:
        return raw

    first_line = raw.strip().split("\n")[0].strip()
    field_type = field.get("type", "")

    # ── radio / select / conditional_select ──────────────────────────────────
    if field_type in ("radio", "select", "conditional_select"):
        options = field.get("options", [])
        if isinstance(options, dict):
            # Flatten conditional_select options dict
            flat = []
            for sub_list in options.values():
                flat.extend(sub_list)
            options = flat
        # Exact whole-word match on first line
        for opt in options:
            if re.search(rf"\b{re.escape(str(opt))}\b", first_line, re.IGNORECASE):
                return str(opt)
        # Broader match in full response
        for opt in options:
            if re.search(rf"\b{re.escape(str(opt))}\b", raw, re.IGNORECASE):
                return str(opt)
        return first_line

    # ── conditional ──────────────────────────────────────────────────────────
    elif field_type == "conditional":
        line_lower = first_line.lower()
        if line_lower.startswith("yes") or re.search(r"\byes\b", line_lower):
            return "Yes"
        if line_lower.startswith("no") or re.search(r"\bno\b", line_lower):
            return "No"
        return first_line

    # ── slider ───────────────────────────────────────────────────────────────
    elif field_type == "slider":
        min_v = field.get("min", 0)
        max_v = field.get("max", 10)
        # Try to find a number in valid range on first line
        match = re.search(r"\b(\d+)\b", first_line)
        if match:
            val = int(match.group(1))
            if min_v <= val <= max_v:
                return str(val)
        # Broader search
        for m in re.finditer(r"\b(\d+)\b", raw):
            val = int(m.group(1))
            if min_v <= val <= max_v:
                return str(val)
        return first_line

    # ── number ───────────────────────────────────────────────────────────────
    elif field_type == "number":
        min_v = field.get("min", None)
        max_v = field.get("max", None)
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", first_line)
        if not match:
            match = re.search(r"\b(\d+(?:\.\d+)?)\b", raw)
        if match:
            val_str = match.group(1)
            val = float(val_str)
            if min_v is not None and max_v is not None:
                if min_v <= val <= max_v:
                    return val_str
            else:
                return val_str
        return first_line

    # ── frequency ────────────────────────────────────────────────────────────
    elif field_type == "frequency":
        # First: check if LLM returned a canonical value directly
        candidate = first_line.lower().strip()
        if candidate in FREQUENCY_COLUMN_MAP:
            return FREQUENCY_COLUMN_MAP[candidate]
        # Second: try to find a canonical value anywhere in the response
        for line in raw.split("\n"):
            cleaned = line.lower().strip()
            if cleaned in FREQUENCY_COLUMN_MAP:
                return FREQUENCY_COLUMN_MAP[cleaned]
        # Third: phrase matching against the full response
        text_lower = raw.lower()
        for pattern, canonical in FREQUENCY_PHRASE_MAP:
            if re.search(pattern, text_lower):
                return canonical
        return first_line

    # ── textarea ─────────────────────────────────────────────────────────────
    elif field_type == "textarea":
        # Return first meaningful paragraph, up to 200 chars
        paragraphs = [p.strip() for p in raw.strip().split("\n\n") if p.strip()]
        if paragraphs:
            text = paragraphs[0]
            return text[:200] if len(text) > 200 else text
        return first_line[:200]

    # ── time_range ───────────────────────────────────────────────────────────
    elif field_type == "time_range":
        # Look for HH:MM–HH:MM or similar patterns
        match = re.search(r"\d{1,2}:\d{2}\s*[–\-to]+\s*\d{1,2}:\d{2}", raw)
        if match:
            return match.group(0).strip()
        # Also accept plain hour ranges like "8–10 am"
        match = re.search(r"\d{1,2}\s*(?:am|pm)\s*[–\-to]+\s*\d{1,2}\s*(?:am|pm)", raw, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        return first_line

    # ── fallback ─────────────────────────────────────────────────────────────
    return first_line
