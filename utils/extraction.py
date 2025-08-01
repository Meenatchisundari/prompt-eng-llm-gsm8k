import re

def extract_answer_number(text):
    
    patterns = [
        r'(?i)the answer is\s*[:\-]?\s*\$?([+-]?\d+(?:\.\d+)?)',
        r'(?i)answer\s*[:\-]?\s*\$?([+-]?\d+(?:\.\d+)?)',
        r'(?i)=\s*\$?([+-]?\d+(?:\.\d+)?)',
        r'####\s*\$?([+-]?\d+(?:\.\d+)?)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    
    # Fallback: pick the last number in the output
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None

