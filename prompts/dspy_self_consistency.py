import dspy

class SelfConsistencyDSPy(dspy.Module):
    def __init__(self, num_samples=5):
        super().__init__()
        self.module = dspy.ChainOfThought(CoTDSPy(), num_generations=num_samples)

    def forward(self, question):
        return self.module(question=question)
