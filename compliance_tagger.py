
def detect_compliance_issues(text: str) -> list:
    """Flag known risky compliance terms."""
    risky_terms = [
        "restatement",
        "earnings risk",
        "regulatory breach",
        "non-compliance",
        "violation",
        "penalty",
        "audit failure"
    ]
    
    flagged = [term for term in risky_terms if term.lower() in text.lower()]
    return flagged
