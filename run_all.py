import sys
import os
import csv
from datetime import datetime

exec(open("setup_path.py").read())
from utils.self_consistency_utils import majority_vote
from utils.evaluation import evaluate_local_model
from utils.extraction import extract_answer_number
from models.llama2_loader import load_llama2_quantized
from models.qwen_loader import load_qwen_quantized

from prompts.zero_shot import zero_shot_prompt
from prompts.cot import cot_prompt
from prompts.few_shot import few_shot_prompt
from prompts.self_consistency import self_consistency_prompt
from prompts.prolog_style import prolog_prompt

# Register all available strategies
PROMPT_FUNCTIONS = {
    "zero_shot": zero_shot_prompt,
    "cot": cot_prompt,
    "few_shot": few_shot_prompt,
    "self_consistency": self_consistency_prompt,
    "prolog": prolog_prompt
}

# Register model loaders
MODEL_LOADERS = {
    "llama": load_llama2_quantized,
    "qwen": load_qwen_quantized
}

def run_all(model_name: str, sample_size: int = 20):
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(MODEL_LOADERS)}")

    generator = MODEL_LOADERS[model_name]()
    all_results = []

    for strategy_name, prompt_fn in PROMPT_FUNCTIONS.items():
        print(f"\n Running {strategy_name.replace('_', ' ').title()} on {model_name.upper()} ({sample_size} samples)")
        
        df = evaluate_local_model(
            model_name=model_name,
            generator=generator,
            strategy_name=strategy_name,
            prompt_fn=prompt_fn,
            num_problems=sample_size,
            log_incorrect=True
        )
        all_results.extend(df.to_dict("records"))

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/results_{model_name}_{sample_size}_{timestamp}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "strategy", "question", "correct_answer", "predicted_answer", "correct", "time_taken"])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n All strategies completed. Results saved to: {filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GSM8K evaluation on a selected model.")
    parser.add_argument("model", type=str, help="Model name: one of [llama, qwen]")
    parser.add_argument("samples", type=int, help="Number of GSM8K problems to evaluate")

    args = parser.parse_args()
    run_all(model_name=args.model.lower(), sample_size=args.samples)

