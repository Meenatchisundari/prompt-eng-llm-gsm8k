import dspy

class GSM8KSignature(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Final numeric answer in format: #### [answer]")

class ZeroShotDSPy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(GSM8KSignature)

    def forward(self, question):
        return self.predict(question=question)
