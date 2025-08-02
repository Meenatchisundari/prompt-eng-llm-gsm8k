import dspy

class ZeroShotSignature(dspy.Signature):
    """Basic QA without CoT."""
    question = dspy.InputField(desc="A math word problem")
    answer = dspy.OutputField(desc="Final answer as a number")

class ZeroShotDSPy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ZeroShotSignature)

    def forward(self, question):
        return self.predict(question=question)
