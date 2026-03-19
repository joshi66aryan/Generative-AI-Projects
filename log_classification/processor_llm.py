from dotenv import load_dotenv
from groq import Groq

load_dotenv()
_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq()
    return _groq_client

def classify_with_llm(log_message):
    prompt = f"""Classify the log message into one of these categories:
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category return "Unclassified".
    Only return the category name. No preamble.
    Log message:{log_message}"""

    allowed = {"Workflow Error", "Deprecation Warning", "Unclassified"}
    try:
        client = _get_groq_client()
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = (chat_completion.choices[0].message.content or "").strip()
    except Exception:
        return "Unclassified"

    candidate = content.strip().strip("\"'").splitlines()[0].strip()
    return candidate if candidate in allowed else "Unclassified"


if __name__ == "__main__":
    print(classify_with_llm(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classify_with_llm(
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(classify_with_llm("System reboot initiated by user 12345."))
