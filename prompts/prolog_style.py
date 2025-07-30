def prolog_prompt(question: str) -> str:
    return (
        f"% Solve using logical reasoning and facts\n"
        f"% Problem: {question}\n\n"
        f"% Facts (extract given information):\n"
        f"% Rules (define relationships):\n"
        f"% Query (what we need to find):\n"
        f"% Answer:"
    )
