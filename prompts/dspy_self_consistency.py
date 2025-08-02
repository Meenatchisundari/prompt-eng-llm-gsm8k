import dspy

class SelfConsistencySignature(dspy.Signature):
    question = dspy.InputField(desc="Math problem requiring step-by-step reasoning")
    answer = dspy.OutputField(desc="Final numeric answer in format #### [answer]")

class SelfConsistencyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SelfConsistencySignature)

    def forward(self, question):
        prompt = (
            f"Solve this math problem step by step, showing your reasoning clearly.\n\n"
            f"Problem: {question}\n\n"
            f"Let me work through this step by step:"
        )
        return self.predict(question=prompt)
