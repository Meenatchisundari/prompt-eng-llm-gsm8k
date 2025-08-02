import dspy

class SelfConsistencyDSPy(dspy.Module):
    def __init__(self, n=5):
        super().__init__()
        self.sc = dspy.ChainOfThought(CoTDSPy(), num_generations=n)

    def forward(self, question):
        return self.sc(question=question)
