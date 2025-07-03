import re

def detect_pii(text: str) -> dict:
    """Detect PII in the given text."""
    pii = {
        "emails": re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', text),
        "phones": re.findall(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', text),
        "ssns": re.findall(r'\b\d{3}-\d{2}-\d{4}\b', text)
    }
    return {k: v for k, v in pii.items() if v}
