import dspy

class PrologSignature(dspy.Signature):
    question = dspy.InputField(desc="Logic-based math problem")
    answer = dspy.OutputField(desc="Answer stated clearly")

class PrologModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PrologSignature)

    def forward(self, question):
        prompt = (
            f"% Solve using logical reasoning and facts\n"
            f"% Problem: {question}\n\n"
            f"% Facts (extract given information):\n"
            f"% Rules (define relationships):\n"
            f"% Query (what we need to find):\n"
            f"% Answer:"
        )
        return self.predict(question=prompt)
