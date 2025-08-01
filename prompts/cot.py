def cot_prompt(question: str) -> str:
    return (
        '''f"Let's work through this step by step.\n\n"
        f"Question: {question}\n\n"
        f"Let's think step by step:\n"'''
        f"Let's work through this step by step. Give only the final numeric answer at the end in the format: #### [answer].\n\nQuestion: {question}\nLet's think step by step:"

    )
    
