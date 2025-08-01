def zero_shot_prompt(question):
    return f"""You are a math tutor. Solve the problem and give only the final numeric answer in the format: #### [answer].\n\nQuestion: {question}\nAnswer:"""
