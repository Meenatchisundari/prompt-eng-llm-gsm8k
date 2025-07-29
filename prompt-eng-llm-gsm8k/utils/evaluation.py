from prompts.zero_shot import zero_shot_prompt
from models.gpt4_loader import query_gpt4
from utils.extraction import extract_answer_number
from datasets import load_dataset
import pandas as pd
import time

def evaluate_strategy(strategy_name, prompt_fn, num_problems=10):
    dataset = load_dataset("gsm8k", "main")["test"]
    results = []

    for i in range(num_problems):
        question = dataset[i]["question"]
        correct = extract_answer_number(dataset[i]["answer"])
        prompt = prompt_fn(question)
        start = time.time()
        response = query_gpt4(prompt)
        prediction = extract_answer_number(response)
        duration = round(time.time() - start, 2)

        results.append({
            "strategy": strategy_name,
            "question": question,
            "correct_answer": correct,
            "predicted_answer": prediction,
            "correct": prediction == correct,
            "time_taken": duration
        })

    return pd.DataFrame(results)
