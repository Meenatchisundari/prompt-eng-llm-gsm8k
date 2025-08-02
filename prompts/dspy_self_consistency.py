class SelfConsistencyDSPy(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Final answer as #### number.")

class SelfConsistencyModule(dspy.ChainOfThought):
    def __init__(self, n=5):
        super().__init__(SelfConsistencyDSPy, n=n)

    def forward(self, question):
        return self(question=question)
