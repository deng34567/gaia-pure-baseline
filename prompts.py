from __future__ import annotations

SYSTEM_PROMPT = """You are solving one GAIA benchmark question.
This run is a deliberately weak single-model baseline.
You cannot browse the web, access attachments, read files, run code, or use any external tools.
Use only the question text shown to you.

Output exactly one line in this format:
FINAL ANSWER: <answer>

Rules:
- Do not output square brackets or placeholder text.
- Do not explain your reasoning.
- Do not restate the question.
- Do not describe your limitations or mention tools.
- If the answer is a number, output only the number unless the question explicitly requires units or formatting.
- If the answer is a string, keep it as short as possible.
- If you are unsure, still give your single best guess.
"""

ATTACHMENT_WARNING = (
    "This task may reference an attachment, but you cannot access attachments or external tools in this run. "
    "Do not mention this limitation in your answer. Use only the visible question text and still provide one best-guess answer."
)


def build_messages(question: str, has_attachment: bool) -> list[dict[str, str]]:
    user_content = question.strip()
    if has_attachment:
        user_content += "\n\n" + ATTACHMENT_WARNING
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
