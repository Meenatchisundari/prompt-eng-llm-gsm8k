import dspy

class CoTSignature(dspy.Signature):
    question = dspy.InputField(desc="Step-by-step math problem")
    answer = dspy.OutputField(desc="Final numeric answer in the format: #### [answer]")

class CoTModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CoTSignature)

    def forward(self, question):
        prompt = (
            f"Let's work through this step by step. Give only the final numeric answer at the end "
            f"in the format: #### [answer].\n\nQuestion: {question}\nLet's think step by step:"
        )
        return self.predict(question=prompt)
