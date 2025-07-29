def zero_shot_prompt(question):
    return f"""You are a math tutor. Solve the following problem and give the final answer as a number.

{question}

Answer:"""
