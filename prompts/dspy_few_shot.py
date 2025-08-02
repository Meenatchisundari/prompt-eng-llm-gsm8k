import dspy

FEW_SHOT_EXAMPLES = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: I start with 15 trees and end with 21 trees. So 21 - 15 = 6. #### 6

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 and ended with 12. So 20 - 12 = 8. #### 8
"""

class FewShotDSPy(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Final numeric answer in format: #### [answer]")

class FewShotModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(FewShotDSPy)

    def forward(self, question):
        prompt = f"{FEW_SHOT_EXAMPLES}\nQ: {question}\nA:"
        return self.predict(question=prompt)

FewShotDSPy = FewShotModule()
