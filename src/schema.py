"""Single source of truth for the task: fields, JSON schema and the prompt.

Training, evaluation and serving all import from here, so the model is always
evaluated with exactly the prompt it was trained on.
"""

import json

FIELDS = ["company", "date", "address", "total"]

# JSON schema used for (a) validating model output and (b) vLLM structured outputs.
RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {f: {"type": ["string", "null"]} for f in FIELDS},
    "required": FIELDS,
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You extract key information from the OCR text of a scanned receipt.\n"
    "Return ONLY a JSON object with exactly these keys: "
    '"company", "date", "address", "total".\n'
    "- company: the name of the shop or company that issued the receipt\n"
    "- date: the transaction date, copied as printed\n"
    "- address: the address of the issuing company\n"
    "- total: the final amount paid as printed, keeping the decimal point "
    "(e.g. 9.00 or 127.20), without a currency symbol\n"
    "Use null if a field is not present."
)


def build_prompt(ocr_text: str) -> list[dict]:
    """Chat messages for one receipt (everything except the answer)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Receipt OCR text:\n\n{ocr_text}"},
    ]


def to_target(gold: dict) -> str:
    """Canonical JSON string the model is trained to produce (fixed key order)."""
    return json.dumps({f: gold.get(f) for f in FIELDS}, ensure_ascii=False)
