import dspy

class PrologDSPy(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Final numeric answer using logic: #### [answer]")

class PrologModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PrologDSPy)

    def forward(self, question):
        prompt = f"""% Solve using logical reasoning and facts
% Problem: {question}

% Facts (extract given information):
% Rules (define relationships):
% Query (what we need to find):
% Answer:"""
        return self.predict(question=prompt)

PrologDSPy = PrologModule()
