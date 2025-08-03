import sys
import os
import csv
import argparse
import dspy
import random
from datetime import datetime

exec(open("setup_path.py").read())

from prompts.dspy_zero_shot import ZeroShotModule
from prompts.dspy_cot import CoTModule
from prompts.dspy_few_shot import FewShotModule
from prompts.dspy_prolog import PrologModule
from prompts.dspy_self_consistency import SelfConsistencyModule

from utils.extraction import extract_answer_number
from utils.dataset import download_gsm8k_dataset
from models.llama2_dspy_loader import load_llama2_dspy
from models.qwen_dspy_loader import load_qwen_dspy


STRATEGIES = {
    "zero_shot": ZeroShotModule,
    "cot": CoTModule,
    "few_shot": FewShotModule,
    "prolog": PrologModule,
    "self_consistency": SelfConsistencyModule,
}

MODEL_LOADERS = {
    "llama": load_llama2_dspy,
    "qwen": load_qwen_dspy,
}


def evaluate_dspy(strategy_name, module_class, dataset, samples=1):
    results = []
    correct = 0
    module = module_class()

    for i, sample in enumerate(dataset):
        question = sample["question"]
        gt = extract_answer_number(sample["answer"])

        try:
            if strategy_name == "self_consistency":
                answers = []
                for _ in range(samples):
                    response = module(question=question).answer
                    answers.append(extract_answer_number(response))
                pred = max(set(answers), key=answers.count) if answers else None
            else:
                response = module(question=question).answer
                pred = extract_answer_number(response)
        except Exception as e:
            print(f"[ERROR] Q{i+1}: {e}")
            continue

        is_correct = pred == gt
        correct += int(is_correct)

        print(f"\nQ{i+1}: {question}\nPredicted: {pred}, GT: {gt}, Correct: {is_correct}")
        results.append({
            "strategy": strategy_name,
            "question": question,
            "correct_answer": gt,
            "predicted_answer": pred,
            "correct": is_correct
        })

    if results:
        acc = correct / len(results)
        print(f"\n {strategy_name} Accuracy: {acc*100:.2f}%")
    else:
        print(f"\n No valid predictions for {strategy_name}")
    return results


def run_all_dspy(model_name, sample_size):
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Invalid model: {model_name}. Choose from {list(MODEL_LOADERS)}")

    lm = MODEL_LOADERS[model_name]()
    dspy.configure(lm=lm)

    dataset = download_gsm8k_dataset()["test"]
    samples = random.sample(dataset, sample_size)
    all_results = []

    for name, module_class in STRATEGIES.items():
        print(f"\n=== Running DSPy: {name} ===")
        result = evaluate_dspy(name, module_class, samples, samples=5 if name == "self_consistency" else 1)
        all_results.extend(result)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/dspy_results_{model_name}_{sample_size}_{timestamp}.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "question", "correct_answer", "predicted_answer", "correct"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DSPy strategies on GSM8K")
    parser.add_argument("model", type=str, choices=["llama", "qwen"])
    parser.add_argument("samples", type=int)
    args = parser.parse_args()

    run_all_dspy(args.model, args.samples)
