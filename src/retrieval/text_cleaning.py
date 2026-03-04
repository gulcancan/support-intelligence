"""
Text cleaning for dual-representation retrieval.

Strategy:
  - BM25 uses ORIGINAL text (preserves exact error codes, version numbers, tokens)
  - Vector search uses CLEANED text (strips noise for better semantic embeddings)
  - Resolution summaries are cleaned at INDEX TIME (one-time cost, not per query)

Cleaning operations (lightweight, no LLM):
  1. Strip HTML tags and email signatures
  2. Remove quoted replies ("On Jan 1, John wrote: ...")
  3. Remove excessive whitespace, special chars, repeated punctuation
  4. Collapse multi-line stack traces to first+last line (preserve error, drop frames)
  5. Normalize product/version mentions
  6. Strip greeting/closing boilerplate ("Hi team,", "Best regards,")
"""
import re
import logging

logger = logging.getLogger(__name__)

# ── Greeting / closing patterns ────────────────────────────────────────────
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|dear|good\s+(morning|afternoon|evening)|greetings)[\s,!.]*"
    r"(team|support|everyone|all|there)?[\s,!.]*\n?",
    re.IGNORECASE | re.MULTILINE,
)
_CLOSING_RE = re.compile(
    r"(best\s+regards|kind\s+regards|regards|thanks|thank\s+you|cheers|sincerely|"
    r"many\s+thanks|thanks\s+in\s+advance|looking\s+forward|please\s+advise|"
    r"appreciate\s+your\s+help)[,.\s!]*$",
    re.IGNORECASE | re.MULTILINE,
)

# ── Quoted reply pattern ───────────────────────────────────────────────────
_QUOTED_REPLY_RE = re.compile(
    r"(on\s+.{5,60}\s+wrote\s*:|from\s*:\s*.+\nsent\s*:\s*.+\nto\s*:\s*.+\nsubject\s*:\s*.+\n|"
    r"^>.*$|"
    r"-{3,}\s*original\s+message\s*-{3,}|"
    r"-{3,}\s*forwarded\s+message\s*-{3,})",
    re.IGNORECASE | re.MULTILINE,
)

# ── HTML / email boilerplate ───────────────────────────────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_EMAIL_HEADER_RE = re.compile(
    r"^(from|to|cc|bcc|sent|date|subject)\s*:\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)

# ── Stack trace collapse ──────────────────────────────────────────────────
_STACK_FRAME_RE = re.compile(
    r"(^\s*at\s+.+|^\s*File\s+\".+\",\s+line\s+\d+|^\s*#\d+\s+0x[0-9a-fA-F]+)",
    re.MULTILINE,
)

# ── Noise patterns ─────────────────────────────────────────────────────────
_REPEATED_PUNCT_RE = re.compile(r"([!?.=\-]){3,}")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
_MULTIPLE_SPACES_RE = re.compile(r" {2,}")


def clean_ticket_text(subject: str, description: str) -> str:
    """
    Clean ticket text for semantic embedding (vector search).

    Strips noise while preserving the core issue description.
    Returns a single cleaned string: "subject. description"
    """
    subject = (subject or "").strip()
    desc = (description or "").strip()

    # Strip HTML
    desc = _HTML_TAG_RE.sub(" ", desc)

    # Remove email headers
    desc = _EMAIL_HEADER_RE.sub("", desc)

    # Remove quoted replies (everything after the quote marker)
    match = _QUOTED_REPLY_RE.search(desc)
    if match:
        desc = desc[:match.start()].strip()

    # Remove greetings and closings
    desc = _GREETING_RE.sub("", desc)
    desc = _CLOSING_RE.sub("", desc)

    # Collapse stack traces: keep first and last line only
    lines = desc.split("\n")
    stack_lines = []
    non_stack_lines = []
    in_stack = False
    for line in lines:
        if _STACK_FRAME_RE.match(line):
            stack_lines.append(line.strip())
            in_stack = True
        else:
            if in_stack and stack_lines:
                # Emit first + last stack frame
                if len(stack_lines) > 2:
                    non_stack_lines.append(stack_lines[0])
                    non_stack_lines.append(f"  ... ({len(stack_lines)-2} more frames)")
                    non_stack_lines.append(stack_lines[-1])
                else:
                    non_stack_lines.extend(stack_lines)
                stack_lines = []
                in_stack = False
            non_stack_lines.append(line)
    # Flush any remaining stack
    if stack_lines:
        if len(stack_lines) > 2:
            non_stack_lines.append(stack_lines[0])
            non_stack_lines.append(f"  ... ({len(stack_lines)-2} more frames)")
            non_stack_lines.append(stack_lines[-1])
        else:
            non_stack_lines.extend(stack_lines)

    desc = "\n".join(non_stack_lines)

    # Normalize whitespace
    desc = _REPEATED_PUNCT_RE.sub(r"\1", desc)
    desc = _MULTIPLE_NEWLINES_RE.sub("\n\n", desc)
    desc = _MULTIPLE_SPACES_RE.sub(" ", desc)
    desc = desc.strip()

    # Combine: subject provides concise summary, description provides detail
    if subject and desc:
        return f"{subject}. {desc}"
    return subject or desc


def clean_resolution_text(resolution: str) -> str:
    """
    Clean resolution text for indexing.

    This runs at INDEX TIME (one-time cost per ticket), not per-query.
    Strips boilerplate from agent responses, keeps actionable content.
    """
    text = (resolution or "").strip()
    if not text:
        return text

    # Strip HTML
    text = _HTML_TAG_RE.sub(" ", text)

    # Remove agent greetings/closings
    text = _GREETING_RE.sub("", text)
    text = _CLOSING_RE.sub("", text)

    # Remove quoted customer text in replies
    match = _QUOTED_REPLY_RE.search(text)
    if match:
        text = text[:match.start()].strip()

    # Normalize whitespace
    text = _MULTIPLE_NEWLINES_RE.sub("\n\n", text)
    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    text = text.strip()

    return text
