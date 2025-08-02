import dspy

class ZeroShotDSPy(dspy.Signature):
    """Zero-shot DSPy signature for GSM8K."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Final numeric answer in format: #### [answer]")

class ZeroShotModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ZeroShotDSPy)

    def forward(self, question):
        prompt = f"You are a math tutor. Solve the problem and give only the final numeric answer in the format: #### [answer].\n\nQuestion: {question}\nAnswer:"
        return self.predict(question=prompt)

# Instantiate for run_all_dspy
ZeroShotDSPy = ZeroShotModule
