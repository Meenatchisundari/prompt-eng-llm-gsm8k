import dspy
import random

class SelfConsistencyDSPy(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Final numeric answer in format: #### [answer]")

class SelfConsistencyModule(dspy.Module):
    def __init__(self, num_samples=5):
        super().__init__()
        self.num_samples = num_samples
        self.predict = dspy.Predict(SelfConsistencyDSPy)

    def forward(self, question):
        answers = []
        for _ in range(self.num_samples):
            prompt = f"Solve this math problem step by step.\n\nProblem: {question}\n\nLet me work through this step by step:"
            out = self.predict(question=prompt)
            answers.append(out.answer)
        # Naive majority vote
        return max(set(answers), key=answers.count)

SelfConsistencyDSPy = SelfConsistencyModule()
