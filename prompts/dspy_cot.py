import dspy

class CoTSignature(dspy.Signature):
    question = dspy.InputField(desc="Math problem")
    answer = dspy.OutputField(desc="Reasoned answer ending with #### [answer]")

class CoTDSPy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CoTSignature)

    def forward(self, question):
        return self.predict(question=question)
