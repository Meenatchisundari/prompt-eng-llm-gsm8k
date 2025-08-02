import dspy

class CoTSignature(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Step-by-step reasoning followed by final answer")

class CoTDSPy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CoTSignature)

    def forward(self, question):
        return self.predict(question=question)
