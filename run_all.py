import sys
exec(open("setup_path.py").read())

from utils.dataset import download_gsm8k_dataset
from prompts.zero_shot import zero_shot_prompt

def run_zero_shot(gsm8k, generator, model_name):
    correct = 0
    total = len(gsm8k['test'])

    print(f"\nRunning Zero-Shot Inference on {total} GSM8K test problems using {model_name}...\n")

    for i, sample in enumerate(gsm8k['test'][:5]):  # Limit to 5 for demo
        question = sample['question']
        answer = sample['answer']

        prompt = zero_shot_prompt(question)
        output = generator(prompt)[0]['generated_text']
        predicted = output.split("Answer:")[-1].strip()

        print(f"Q{i+1}: {question}")
        print(f"Expected: {answer}")
        print(f"Generated: {predicted}")
        print("-" * 50)

    print(f"\nZero-shot inference complete with {model_name}.")

def load_model(model_name):
    if model_name.lower() == "llama":
        from models.llama2_loader import load_llama2_quantized
        return load_llama2_quantized()
    elif model_name.lower() == "qwen":
        from models.qwen_loader import load_qwen_quantized
        return load_qwen_quantized()
    else:
        raise ValueError(f"Unsupported model: {model_name}. Use 'llama' or 'qwen'.")

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "llama"

    gsm8k = download_gsm8k_dataset()
    if gsm8k is None:
        print("Dataset download failed.")
        exit()

    generator = load_model(model_name)
    run_zero_shot(gsm8k, generator, model_name)
