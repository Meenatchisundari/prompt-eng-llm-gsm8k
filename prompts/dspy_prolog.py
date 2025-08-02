class PrologDSPy(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Solve using logic reasoning. Answer as #### number.")

class PrologModule(dspy.Predict):
    def __init__(self):
        super().__init__()
        self.program = dspy.ChainOfThought(PrologDSPy)

    def forward(self, question):
        return self.program(question=question)
