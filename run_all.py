from utils.dataset import download_gsm8k_dataset
from models.llama2_loader import load_llama2_quantized
from prompts.zero_shot import zero_shot_prompt

def run_zero_shot_llama(gsm8k, generator):
    correct = 0
    total = len(gsm8k['test'])

    print(f"\nRunning Zero-Shot Inference on {total} GSM8K test problems...")

    for i, sample in enumerate(gsm8k['test'][:5]):  # Limit to 5 for demo
        question = sample['question']
        answer = sample['answer']

        prompt = zero_shot_prompt(question)
        output = generator(prompt)[0]['generated_text']
        predicted = output.split("Answer:")[-1].strip()

        print(f"\nQ{i+1}: {question}")
        print(f"Expected: {answer}")
        print(f"Generated: {predicted}")

    print("\nZero-shot inference complete.")

if __name__ == "__main__":
    # Step 1: Load dataset
    gsm8k = download_gsm8k_dataset()
    if gsm8k is None:
        print("Dataset download failed.")
        exit()

    # Step 2: Load model
    generator = load_llama2_quantized()

    # Step 3: Run strategy
    run_zero_shot_llama(gsm8k, generator)
