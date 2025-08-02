import dspy

class CoTDSPy(dspy.Signature):
    """Chain-of-thought DSPy signature."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Final numeric answer in format: #### [answer]")

class CoTModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CoTDSPy)

    def forward(self, question):
        prompt = f"Let's work through this step by step.\n\nQuestion: {question}\n\nLet's think step by step. Give only the final numeric answer at the end in the format: #### [answer]."
        return self.predict(question=prompt)

CoTDSPy = CoTModule()
