from prompts.zero_shot import zero_shot_prompt
from utils.evaluation import evaluate_strategy

df = evaluate_strategy("zero_shot", zero_shot_prompt, num_problems=10)
df.to_csv("results_zero_shot.csv", index=False)
print(df)
