"""External Final termination guard; never edits logits or generated text."""
import hashlib
import re


def visible_unit(unit):
    """Normalize a paragraph claim while ignoring citation label wording."""
    unit = re.sub(r'(?is)<think>.*?</think>', '', unit)
    unit = re.sub(r'(?is)</?answer\b[^>]*>', '', unit)
    unit = re.sub(r'(?is)\s*<cite\b[^>]*>.*?</cite>', '', unit)
    return re.sub(r'\s+', ' ', unit).strip()


def repetition_event(text):
    """Stop when a long paragraph is generated a second time.

    The repeated paragraph need not be adjacent: whole-answer loops commonly
    repeat a block of several paragraphs.  Exact normalized visible text keeps
    the guard conservative and does not modify logits or accepted text.
    """
    seen = {}
    for index, raw_unit in enumerate(re.split(r"\n\s*\n", text)):
        unit = visible_unit(raw_unit)
        if len(unit) >= 80:
            digest = hashlib.sha256(unit.encode("utf-8")).hexdigest()
            if digest not in seen:
                seen[digest] = index
                continue
            return {
                "stop_reason": "abnormal_repetition",
                "repeat_count": 2,
                "first_index": seen[digest],
                "repeat_index": index,
                "unit_sha256": digest,
                "protocol_valid": False,
                "reward_export_authorized": False,
            }
    return None
