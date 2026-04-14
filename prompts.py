from __future__ import annotations

SYSTEM_PROMPT = """You are a helpful AI assistant solving GAIA benchmark questions.
This run is a deliberately weak single-model baseline.
You cannot browse the web, access attachments, read files, run code, or use any external tools.
Answer using only the question text shown to you.
Return the final answer using as few words as possible.
If the answer is a number, output only the number without units, commas, or extra text unless the task explicitly requires them.
If the answer is a string, keep it concise and do not add explanations, articles, or extra punctuation unless required by the task.
If the answer is a comma separated list, format each item concisely and separate items with commas only.
Finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]"""

ATTACHMENT_WARNING = (
    "This baseline cannot access attachments or external tools. "
    "If the task depends on an attachment or outside information, answer using only the question text."
)


def build_messages(question: str, has_attachment: bool) -> list[dict[str, str]]:
    user_content = question.strip()
    if has_attachment:
        user_content += "\n\n" + ATTACHMENT_WARNING
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
