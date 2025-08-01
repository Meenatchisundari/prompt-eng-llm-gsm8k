'''import sys
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
        writer.writerows(all_results)'''


import sys
import os
import csv
from datetime import datetime
exec(open("setup_path.py").read())
from utils.self_consistency_utils import majority_vote
from utils.evaluation import evaluate_local_model
from utils.extraction import extract_answer_number
from utils.dataset import download_gsm8k_dataset  # Import your dataset loader
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
    """
    Run all prompt strategies on the specified model with GSM8K dataset.
    
    Args:
        model_name: Name of the model to use ('llama' or 'qwen')
        sample_size: Number of problems to evaluate (default: 20)
    """
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(MODEL_LOADERS.keys())}")
    
    # Download and load GSM8K dataset
    print("Loading GSM8K dataset...")
    dataset = download_gsm8k_dataset()
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Get the test problems and limit to sample_size
    problems = dataset['test'][:sample_size]
    print(f"Using {len(problems)} problems for evaluation")
    
    # Load model and tokenizer
    print(f"Loading {model_name.upper()} model...")
    model_components = MODEL_LOADERS[model_name]()
    
    # Handle different return formats from model loaders
    if isinstance(model_components, tuple):
        model, tokenizer = model_components
    else:
        # If only model is returned, you'll need to load tokenizer separately
        model = model_components
        # You'll need to implement tokenizer loading based on your model setup
        raise ValueError("Model loader should return both model and tokenizer")
    
    # Fix the padding token issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    all_results = []
    
    for strategy_name, prompt_fn in PROMPT_FUNCTIONS.items():
        print(f"\n{'='*50}")
        print(f"Running {strategy_name.replace('_', ' ').title()} Strategy")
        print(f"Model: {model_name.upper()} | Problems: {sample_size}")
        print('='*50)
        
        try:
            df = evaluate_local_model(
                model=model,
                tokenizer=tokenizer,
                problems=problems,
                sample_size=sample_size,
                strategy_name=strategy_name,
                prompt_fn=prompt_fn
            )
            
            # Add metadata to results
            for record in df.to_dict("records"):
                record['model'] = model_name
                record['strategy'] = strategy_name
                all_results.append(record)
                
            print(f"✓ {strategy_name} completed successfully")
            
        except Exception as e:
            print(f"✗ Error in {strategy_name}: {str(e)}")
            continue
    
    # Save results to CSV
    if all_results:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/results_{model_name}_{sample_size}_{timestamp}.csv"
        
        # Determine fieldnames from actual data
        if all_results:
            fieldnames = list(all_results[0].keys())
        else:
            fieldnames = ["model", "strategy", "question", "correct_answer", "predicted_answer", "correct", "time_taken"]
        
        with open(filename, "w", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n Results saved to: {filename}")
        print(f" Total evaluations: {len(all_results)}")
        
        # Print summary statistics
        if 'correct' in fieldnames:
            correct_count = sum(1 for r in all_results if r.get('correct', False))
            accuracy = correct_count / len(all_results) * 100
            print(f" Overall accuracy: {accuracy:.2f}% ({correct_count}/{len(all_results)})")
    else:
        print("  No results to save - all strategies failed")

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "llama"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    print(" Starting GSM8K Evaluation Pipeline")
    print(f"Model: {model.upper()}")
    print(f"Sample size: {sample_size}")
    print("-" * 50)
    
    try:
        run_all(model.lower(), sample_size)
    except KeyboardInterrupt:
        print("\n  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n Evaluation failed: {str(e)}")
        raise
