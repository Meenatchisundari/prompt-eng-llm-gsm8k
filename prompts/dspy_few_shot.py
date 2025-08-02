class FewShotDSPy(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Final answer as #### number.")

FEW_SHOT_EXAMPLES = [
    {"question": "Jason had 20 lollipops. Now he has 12. How many did he give?", "answer": "#### 8"},
    {"question": "There are 15 trees. Workers plant more and now there are 21. How many were planted?", "answer": "#### 6"},
]

class FewShotModule(dspy.Predict):
    def __init__(self):
        super().__init__()
        self.program = dspy.Predict(FewShotDSPy).compile(demos=FEW_SHOT_EXAMPLES)

    def forward(self, question):
        return self.program(question=question)
