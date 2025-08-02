import dspy

class ZeroShotSignature(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

class ZeroShotModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ZeroShotSignature)

    def forward(self, question):
        return self.predict(question=question)
