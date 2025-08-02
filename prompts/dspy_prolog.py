import dspy

class PrologSignature(dspy.Signature):
    question = dspy.InputField(desc="Math word problem to convert into logic facts")
    answer = dspy.OutputField(desc="Facts, rules, query, and answer reasoning")

class PrologDSPy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PrologSignature)

    def forward(self, question):
        return self.predict(question=question)
