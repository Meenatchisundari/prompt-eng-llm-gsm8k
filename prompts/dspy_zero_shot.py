import dspy

class ZeroShotSignature(dspy.Signature):
    question = dspy.InputField(desc="Math problem")
    answer = dspy.OutputField(desc="Answer in format: #### [answer]")

class ZeroShotDSPy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ZeroShotSignature)

    def forward(self, question):
        return self.predict(question=question)
