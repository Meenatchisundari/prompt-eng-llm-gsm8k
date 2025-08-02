class ZeroShotDSPy(dspy.Signature):
    """Predict answer with no examples."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Final answer as #### number.")

class ZeroShotModule(dspy.Predict):
    def __init__(self):
        super().__init__()
        self.zero = dspy.ChainOfThought(ZeroShotDSPy)

    def forward(self, question):
        return self.zero(question=question)
