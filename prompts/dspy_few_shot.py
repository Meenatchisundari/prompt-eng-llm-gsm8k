import dspy

class FewShotSignature(dspy.Signature):
    question = dspy.InputField(desc="A math problem")
    answer = dspy.OutputField(desc="Answer with final number")

class FewShotDSPy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(FewShotSignature, demos=[
            FewShotSignature(question="Alice had 10 apples and ate 3. How many are left?", answer="#### 7"),
            FewShotSignature(question="John bought 4 pens and then 2 more. How many total?", answer="#### 6")
        ])

    def forward(self, question):
        return self.predict(question=question)
