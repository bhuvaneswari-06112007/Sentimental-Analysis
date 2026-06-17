import re

# Keyword lists for urgency detection
HIGH_URGENCY_KEYWORDS = {
    'urgent', 'asap', 'immediate', 'critical', 'outage', 'down', 'breach',
    'overdue', 'failure', 'broken', 'emergency', 'blocker', 'blocking', 'payroll'
}

MEDIUM_URGENCY_KEYWORDS = {
    'request', 'pending', 'review', 'reminder', 'onboarding', 'schedule',
    'meeting', 'submit', 'action', 'update', 'task', 'over'
}

def detect_priority(subject: str, body: str) -> dict:
    """
    Rule-based priority detection engine.
    Analyzes subject and body text to determine email priority: 'Low', 'Medium', or 'High'.
    Returns a dictionary with:
    - 'priority': The determined priority ('Low', 'Medium', 'High')
    - 'reasons': A list of reasons for the priority assignment.
    """
    subject = subject.lower()
    body = body.lower()
    combined = f"{subject} {body}"
    
    reasons = []
    high_matches = []
    med_matches = []
    
    # 1. Check for specific high priority indicators
    # Check for [URGENT] or similar tags in the subject
    if re.search(r'\[\s*urgent\s*\]', subject) or re.search(r'\burgent\b', subject):
        reasons.append("Urgent label or keyword in Subject")
        high_matches.append("urgent")
        
    # 2. Check for keywords
    # Tokenize text
    words = set(re.findall(r'\b[a-z]{3,}\b', combined))
    
    for word in words:
        if word in HIGH_URGENCY_KEYWORDS:
            high_matches.append(word)
        elif word in MEDIUM_URGENCY_KEYWORDS:
            med_matches.append(word)
            
    # 3. Check for exclamation marks
    exclamation_count = combined.count('!')
    if exclamation_count >= 3:
        reasons.append(f"High frequency of exclamation marks ({exclamation_count})")
        
    # 4. Check for all caps words (excluding short words like I, A, IT, etc.)
    all_caps_words = re.findall(r'\b[A-Z]{4,}\b', f"{subject} {body}")
    if len(all_caps_words) >= 2:
        reasons.append(f"Multiple capitalized words (urgency indicator): {', '.join(all_caps_words[:3])}")
        
    # Determine priority based on features
    if len(high_matches) >= 2 or (len(high_matches) >= 1 and "Subject" in "".join(reasons)) or exclamation_count >= 3:
        priority = "High"
        if high_matches:
            reasons.append(f"High-priority keywords detected: {', '.join(set(high_matches))}")
    elif len(high_matches) == 1 or len(med_matches) >= 2 or len(all_caps_words) >= 1:
        priority = "Medium"
        if high_matches:
            reasons.append(f"High-priority keyword detected: {high_matches[0]}")
        if med_matches:
            reasons.append(f"Medium-priority keywords detected: {', '.join(set(med_matches[:3]))}")
    else:
        priority = "Low"
        reasons.append("No critical or medium urgency indicators found.")
        
    return {
        'priority': priority,
        'reasons': reasons
    }
