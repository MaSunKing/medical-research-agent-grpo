"""Conservative JSON syntax normalization, never semantic/schema repair."""
from __future__ import annotations

import hashlib
import json
import re

REPAIR_VERSION = "model_json_syntax_v1"
JSON_STAGES = frozenset({"checklist", "decision", "state"})


def _strict_load(text):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key:" + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique_object)


def _syntax_pass(text):
    """Only escape raw LF/CR/TAB within strings and remove trailing commas."""
    out, changes = [], set()
    quoted = escaped = False
    for i, char in enumerate(text):
        if quoted:
            if escaped:
                # A backslash followed by a raw newline is not an unambiguous escape.
                out.append(char)
                escaped = False
            elif char == "\\":
                out.append(char)
                escaped = True
            elif char == '"':
                out.append(char)
                quoted = False
            elif char in "\n\r\t":
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[char])
                changes.add("escape_literal_string_whitespace")
            else:
                out.append(char)
        elif char == '"':
            quoted = True
            out.append(char)
        elif char == ",":
            j = i + 1
            while j < len(text) and text[j] in " \r\n\t":
                j += 1
            previous = next((part.rstrip()[-1] for part in reversed(out) if part.rstrip()), "")
            if j < len(text) and text[j] in "]}" and previous and previous not in "{[,:":
                changes.add("remove_trailing_comma")
            else:
                out.append(char)
        else:
            out.append(char)
    return "".join(out), sorted(changes)


def normalize_model_json(text):
    """Return (effective text, audit or None); uncertain repairs still raise.

    No object extraction from prose, value rewriting, quote/bracket completion,
    key deduplication, missing-comma insertion or single-quote conversion.
    Caller MUST run the original stage/schema/ID validator afterwards.
    """
    try:
        _strict_load(text)
        return text, None
    except json.JSONDecodeError:
        pass
    candidate = text.strip()
    changes = []
    if candidate.startswith("\ufeff"):
        candidate = candidate[1:].strip()
        changes.append("remove_bom")
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", candidate, re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        changes.append("unwrap_json_fence")
    # A single object-only parenthesis envelope (including a stray leading '(').
    if candidate.startswith("("):
        candidate = candidate[1:].strip()
        changes.append("remove_outer_parenthesis")
        if candidate.endswith(")"):
            candidate = candidate[:-1].rstrip()
    # Only an isolated quote AFTER an object, never quotes inside a string.
    if len(candidate) >= 2 and candidate[-1] in ('"', "”", "’") and candidate[-2] == "}":
        candidate = candidate[:-1]
        changes.append("remove_isolated_terminal_quote")
    candidate, syntax_changes = _syntax_pass(candidate)
    changes.extend(syntax_changes)
    value = _strict_load(candidate)  # rejects multiple objects, duplicates, truncation
    if not isinstance(value, dict) or not changes:
        raise ValueError("model_json_no_safe_object_repair")
    audit = dict(version=REPAIR_VERSION, applied=True, changes=changes,
                 original_sha256=hashlib.sha256(text.encode("utf8")).hexdigest(),
                 normalized_sha256=hashlib.sha256(candidate.encode("utf8")).hexdigest(),
                 normalized_text=candidate, model_generated=False)
    return candidate, audit


def parse_model_json(text):
    normalized, _ = normalize_model_json(text)
    return _strict_load(normalized)
