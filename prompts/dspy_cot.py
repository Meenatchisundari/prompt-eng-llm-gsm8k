class CoTDSPy(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Show reasoning step-by-step. Final answer as #### number.")

class CoTModule(dspy.Predict):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(CoTDSPy)

    def forward(self, question):
        return self.cot(question=question)
