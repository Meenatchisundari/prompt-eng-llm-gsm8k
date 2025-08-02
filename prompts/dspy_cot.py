import dspy

class CoTSignature(dspy.Signature):
    """Reasoning before answer."""
    question = dspy.InputField(desc="A math word problem")
    answer = dspy.OutputField(desc="Answer with reasoning and final result")

class CoTDSPy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CoTSignature)

    def forward(self, question):
        return self.predict(question=question)
