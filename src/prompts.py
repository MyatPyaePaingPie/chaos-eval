"""Extraction prompts for different document types."""

# Base extraction prompt - document agnostic
BASE_EXTRACTION_PROMPT = """You are a document extraction specialist. Your task is to extract structured information from the provided document image.

CRITICAL RULES:
1. Extract ONLY what you can clearly see in the document
2. If a field is unclear or missing, use null - DO NOT GUESS
3. If you're uncertain about a value, prefix it with "UNCERTAIN: "
4. Maintain exact spelling/formatting from the document (including errors)
5. Output valid JSON only - no explanations or markdown

Extract all visible fields into a structured JSON object."""

# FUNSD form extraction
FUNSD_EXTRACTION_PROMPT = """You are extracting data from a scanned form document.

The form may contain:
- Header information (titles, dates, reference numbers)
- Questions and answers
- Checkboxes and selections
- Signatures and handwritten notes
- Tables or structured data

Extract ALL visible text organized by its semantic role:
- "header": Document title, form name, dates
- "question": Labels, field names, prompts
- "answer": Filled-in values, responses, data
- "other": Anything else (notes, stamps, signatures)

Return JSON in this format:
{
  "header": ["list of header text"],
  "questions": ["list of question/label text"],
  "answers": ["list of answer/value text"],
  "other": ["list of other text"],
  "key_value_pairs": [
    {"key": "field name", "value": "field value"},
    ...
  ]
}

IMPORTANT: Extract text EXACTLY as it appears, including typos and formatting issues."""

# SROIE receipt extraction
SROIE_EXTRACTION_PROMPT = """You are extracting data from a receipt or invoice.

Extract these specific fields:
- company: The business/store name
- address: Full address of the business
- date: Transaction date (preserve original format)
- total: Total amount (include currency symbol if visible)

Also extract:
- items: List of purchased items with prices if visible
- other_fields: Any other relevant information (tax, subtotal, etc.)

Return JSON in this format:
{
  "company": "string or null",
  "address": "string or null",
  "date": "string or null",
  "total": "string or null",
  "items": [
    {"name": "item name", "price": "price or null"},
    ...
  ],
  "other_fields": {
    "field_name": "value",
    ...
  },
  "confidence": "high|medium|low",
  "issues": ["list any extraction difficulties"]
}

If the receipt is damaged, faded, or partially illegible, note this in "issues"."""

# Generic messy document
CHAOS_EXTRACTION_PROMPT = """You are analyzing a potentially degraded, noisy, or damaged document.

The document may have:
- Faded or low-contrast text
- Noise, artifacts, or damage
- Skewed or rotated content
- Mixed handwritten and printed text
- Unusual formatting

Your task:
1. Extract ALL readable text
2. Preserve the document's structure as best you can
3. Note any areas that are illegible or uncertain

Return JSON:
{
  "extracted_text": "full text content, preserving line breaks",
  "structured_data": {
    // Any key-value pairs you can identify
  },
  "document_type": "form|receipt|letter|report|other|unknown",
  "quality_assessment": {
    "overall": "good|fair|poor|very_poor",
    "issues": ["list of quality issues detected"],
    "illegible_regions": ["descriptions of unreadable areas"]
  },
  "confidence": 0.0-1.0
}"""

# Comparison prompt for hallucination detection
HALLUCINATION_CHECK_PROMPT = """Given this document and this extracted data, verify the extraction:

EXTRACTION TO VERIFY:
{extraction}

For each extracted field, determine:
1. VERIFIED: The value exactly matches what's visible
2. PLAUSIBLE: The value seems reasonable but can't be confirmed
3. HALLUCINATED: The value does NOT appear in the document
4. UNCERTAIN: Can't determine

Return JSON:
{
  "verification": {
    "field_name": "VERIFIED|PLAUSIBLE|HALLUCINATED|UNCERTAIN",
    ...
  },
  "hallucination_detected": true|false,
  "notes": "any relevant observations"
}"""


def get_prompt_for_dataset(dataset_name: str) -> str:
    """Get the appropriate extraction prompt for a dataset."""
    prompts = {
        "funsd": FUNSD_EXTRACTION_PROMPT,
        "sroie": SROIE_EXTRACTION_PROMPT,
        "nod": CHAOS_EXTRACTION_PROMPT,
    }
    return prompts.get(dataset_name.lower(), CHAOS_EXTRACTION_PROMPT)
