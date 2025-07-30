from prompts.cot import cot_prompt

def self_consistency_prompt(question: str) -> str:
    """
    For self-consistency, use this prompt with multiple inference calls
    at temperature 0.7-1.0 to get diverse reasoning paths.
    Then majority vote on the final answers.
    """
    return (
        f"Solve this math problem step by step, showing your reasoning clearly.\n\n"
        f"Problem: {question}\n\n"
        f"Let me work through this step by step:\n"
    )

# Helper function for implementation
def generate_multiple_responses(question: str, model, num_samples: int = 5):
    """
    Generate multiple responses for self-consistency voting.
    Set temperature=0.7-1.0 when calling the model.
    """
    prompt = self_consistency_prompt(question)
    responses = []
    for _ in range(num_samples):
        # Call your model here with temperature > 0
        response = model.generate(prompt, temperature=0.7)
        responses.append(response)
    return responses
