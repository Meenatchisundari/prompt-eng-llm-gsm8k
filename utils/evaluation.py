from utils.self_consistency_utils import majority_vote
from utils.extraction import extract_answer_number
from utils.dataset import download_gsm8k_dataset
import pandas as pd
import time
import random
import os

def evaluate_local_model(model_name, generator, strategy_name, prompt_fn, num_problems=5, log_incorrect=True):
    print(f"\nEvaluating {model_name.upper()} on {num_problems} randomly selected GSM8K problems with strategy: {strategy_name}")

    dataset = download_gsm8k_dataset()["test"]
    samples = random.sample(list(dataset), num_problems)
    results = []

    if log_incorrect:
        os.makedirs("results", exist_ok=True)
        incorrect_log_path = f"results/{model_name}_incorrect_{strategy_name}.txt"
        open(incorrect_log_path, "w").close()  # clear previous

    for i, sample in enumerate(samples):
        question = sample["question"]
        correct = extract_answer_number(sample["answer"])
        prompt = prompt_fn(question)

        start = time.time()

        try:
            response = generator(prompt, max_new_tokens=150, temperature=0.1)[0]['generated_text']
        except Exception as e:
            print(f" Generation failed on Q{i+1}: {e}")
            continue

        duration = round(time.time() - start, 2)
        prediction = extract_answer_number(response)

        print(f"\nQ{i+1}: {question}")
        print(f"Prompt:\n{prompt}")
        print(f"Response:\n{response.strip()}")
        print(f"Expected: {correct}, Predicted: {prediction}")
        print(f"Time Taken: {duration} sec")
        print("-" * 80)

        is_correct = prediction == correct
        results.append({
            "model": model_name,
            "strategy": strategy_name,
            "question": question,
            "correct_answer": correct,
            "predicted_answer": prediction,
            "correct": is_correct,
            "time_taken": duration
        })

        if log_incorrect and not is_correct:
            with open(incorrect_log_path, "a") as f:
                f.write(f"\nQ{i+1}: {question}\n")
                f.write(f"Prompt:\n{prompt}\n")
                f.write(f"Correct: {correct}, Predicted: {prediction}\n")
                f.write(f"Response:\n{response}\n")
                f.write("-" * 80 + "\n")

    return pd.DataFrame(results)
        
