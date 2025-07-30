from collections import Counter
from utils.extraction import extract_answer_number

def majority_vote(responses):
    answers = [extract_answer_number(r) for r in responses]
    answers = [a for a in answers if a is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]
