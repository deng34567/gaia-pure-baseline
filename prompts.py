from __future__ import annotations

SYSTEM_PROMPT = """You are solving one GAIA benchmark question.
This run is a deliberately weak single-model baseline.
You cannot browse the web, access attachments, read files, run code, or use any external tools.
Use only the visible question text.

Your output will be graded automatically.
If you output anything except one final-answer line, it will be counted as wrong.

Valid output examples:
FINAL ANSWER: 4
FINAL ANSWER: Paris
FINAL ANSWER: 12,14,15

Invalid output examples:
I cannot access the web
The answer is probably 4
Reasoning: ...
FINAL ANSWER: The user wants me to solve ...
FINAL ANSWER: Thinking Process: ...
[YOUR FINAL ANSWER]
FINAL ANSWER: <answer>

Rules:
- Output exactly one line.
- The line must start with FINAL ANSWER:
- After FINAL ANSWER:, write only the answer itself.
- Do not explain your reasoning.
- Do not restate the question.
- Do not mention your limitations or unavailable tools.
- Do not describe what the user is asking.
- If you are unsure, still give your single best guess.
"""

ATTACHMENT_WARNING = (
    "This task may reference an attachment, but you cannot access attachments or external tools in this run. "
    "Do not mention this limitation in your answer. Use only the visible question text and still provide one best-guess answer."
)


def build_messages(question: str, has_attachment: bool) -> list[dict[str, str]]:
    user_content = "Question:\n" + question.strip()
    if has_attachment:
        user_content += "\n\n" + ATTACHMENT_WARNING
    user_content += (
        "\n\nReply with exactly one line beginning with FINAL ANSWER: "
        "After the colon, give only the answer itself, not a summary of the question."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
