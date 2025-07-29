import re

def extract_answer_number(text):
    match = re.search(r"(?:####|Answer:)?\\s*([+-]?\\d+(?:\\.\\d+)?)", text)
    return float(match.group(1)) if match else None
