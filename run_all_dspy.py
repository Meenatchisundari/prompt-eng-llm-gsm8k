import sys
import os
import csv
import argparse
import dspy
from datetime import datetime

exec(open("setup_path.py").read())

from prompts.dspy_zero_shot import ZeroShotModule
from prompts.dspy_cot import CoTModule
from prompts.dspy_few_shot import FewShotModule
from prompts.dspy_prolog import PrologModule
from utils.extraction import extract_answer_number
from utils.dataset import download_gsm8k_dataset
from models.llama2_dspy_loader import load_llama2_dspy
from models.qwen_dspy_loader import load_qwen_dspy


# Strategy mapping
STRATEGIES = {
    "zero_shot": ZeroShotModule,
    "cot": CoTModule,
    "few_shot": FewShotModule,
    "prolog": PrologModule
}

# Model loader mapping
MODEL_LOADERS = {
    "llama": load_llama2_dspy,
    "qwen": load_qwen_dspy
}


def evaluate_dspy(strategy_name, module_class, dataset):
    results = []
    correct = 0

    module = module_class()
    for i, sample in enumerate(dataset):
        question = sample["question"]
        gt = extract_answer_number(sample["answer"])

        try:
            output = module(question=question)
            pred = extract_answer_number(output.answer)
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
        print(f"\n [FATAL] No valid predictions for {strategy_name}")
    return results


def run_all_dspy(model_name, sample_size):
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Invalid model: {model_name}. Choose from {list(MODEL_LOADERS)}")

    # Load local model wrapped in DSPy BaseLM
    lm = MODEL_LOADERS[model_name]()
    dspy.configure(lm=lm)

    # Load dataset
    data = download_gsm8k_dataset()["test"][:sample_size]
    all_results = []

    for strategy_name, module_class in STRATEGIES.items():
        print(f"\n=== Running DSPy: {strategy_name} ===")
        result = evaluate_dspy(strategy_name, module_class, data)
        all_results.extend(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    filename = f"results/dspy_results_{model_name}_{sample_size}_{timestamp}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "question", "correct_answer", "predicted_answer", "correct"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n Saved results to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DSPy strategies on GSM8K")
    parser.add_argument("model", type=str, choices=["llama", "qwen"])
    parser.add_argument("samples", type=int)
    args = parser.parse_args()

    run_all_dspy(model_name=args.model, sample_size=args.samples)
