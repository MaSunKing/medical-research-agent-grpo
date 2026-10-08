# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Deployment-shaped context management for the Medical Agent V1 runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


TOOL_PAIR_RE = re.compile(
    r'(?P<call><call_tool\s+name="(?P<name>[^"]+)"[^>]*>.*?</call_tool>\s*)'
    r'(?P<output><tool_output>.*?</tool_output>)',
    re.DOTALL,
)
SNIPPET_RE = re.compile(
    r'(?P<open><snippet\s+id="[^"]+"[^>]*>)(?P<body>.*?)(?P<close></snippet>)',
    re.DOTALL,
)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
SEARCH_TOOLS = {"pubmed_search", "medical_web_search"}
BROWSE_TOOLS = {"browse_document", "browse_webpage"}


@dataclass(frozen=True)
class ContextManagementResult:
    text: str
    tokens: int
    compacted_blocks: int
    over_soft_target: bool


def _compact_snippets(output: str, max_body_chars: int) -> str:
    def replace(match: re.Match[str]) -> str:
        body = " ".join(match.group("body").split())
        if len(body) > max_body_chars:
            body = body[:max_body_chars].rstrip() + " …"
        return match.group("open") + body + match.group("close")

    compacted, count = SNIPPET_RE.subn(replace, output)
    if count:
        return compacted
    # Preserve the protocol tags even for non-snippet tool formats.
    inner = output[len("<tool_output>") : -len("</tool_output>")]
    inner = " ".join(inner.split())
    if len(inner) > max_body_chars:
        inner = inner[:max_body_chars].rstrip() + " …"
    return f"<tool_output>{inner}</tool_output>"


def compact_medical_context(
    text: str,
    *,
    count_tokens: Callable[[str], int],
    soft_limit_tokens: int = 14_000,
    search_snippet_chars: int = 500,
    old_browse_snippet_chars: int = 1_200,
) -> ContextManagementResult:
    """Compact replaceable history while retaining source and citation IDs.

    Search outputs are discovery metadata and are compacted first.  If the
    transcript still exceeds the working target, older browse blocks are
    bounded while the most recent browse remains intact.  The function never
    hard-truncates the user prompt or removes tool/citation identifiers.
    """
    tokens = count_tokens(text)
    if tokens <= soft_limit_tokens:
        return ContextManagementResult(text, tokens, 0, False)

    compacted_blocks = 0

    def compact_search(match: re.Match[str]) -> str:
        nonlocal compacted_blocks
        if match.group("name") not in SEARCH_TOOLS:
            return match.group(0)
        replacement = match.group("call") + _compact_snippets(
            match.group("output"), search_snippet_chars
        )
        if replacement != match.group(0):
            compacted_blocks += 1
        return replacement

    working = TOOL_PAIR_RE.sub(compact_search, text)
    tokens = count_tokens(working)

    if tokens > soft_limit_tokens:
        matches = list(TOOL_PAIR_RE.finditer(working))
        browse_indexes = [
            index
            for index, match in enumerate(matches)
            if match.group("name") in BROWSE_TOOLS
        ]
        newest_browse_index = browse_indexes[-1] if browse_indexes else -1
        pieces = []
        cursor = 0
        for index, match in enumerate(matches):
            pieces.append(working[cursor : match.start()])
            if match.group("name") in BROWSE_TOOLS and index != newest_browse_index:
                replacement = (
                    match.group("call")
                    + _compact_snippets(
                        match.group("output"), old_browse_snippet_chars
                    )
                )
                pieces.append(replacement)
                if replacement != match.group(0):
                    compacted_blocks += 1
            else:
                pieces.append(match.group(0))
            cursor = match.end()
        pieces.append(working[cursor:])
        working = "".join(pieces)
        tokens = count_tokens(working)

    if tokens > soft_limit_tokens:
        think_matches = list(THINK_RE.finditer(working))
        # Keep the latest plan; old simulated reasoning is less important than evidence.
        for match in reversed(think_matches[:-1]):
            working = (
                working[: match.start()]
                + "<think>[earlier plan compacted]</think>"
                + working[match.end() :]
            )
            compacted_blocks += 1
        tokens = count_tokens(working)

    return ContextManagementResult(
        working,
        tokens,
        compacted_blocks,
        tokens > soft_limit_tokens,
    )
