import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# === Load GSM8K ===
def load_sample_data(num_samples=5):
    dataset = load_dataset("gsm8k", "main", split="test")
    return dataset.select(range(num_samples))

# === Answer Extractors ===
def extract_gold_answer(text):
    match = re.search(r"####\s*([+-]?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None

def extract_predicted_answer(response):
    numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', response)
    return float(numbers[-1]) if numbers else None

# === Load Model ===
def load_llama_model():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

# === Evaluate and Debug ===
def run_debug(pipe, dataset):
    correct = 0
    total = len(dataset)

    for i, example in enumerate(dataset):
        question = example['question']
        gold_answer = extract_gold_answer(example['answer'])

        # Minimal prompt (adjust for CoT or few-shot as needed)
        prompt = f"Q: {question}\nA:"
        response = pipe(prompt, max_new_tokens=200, temperature=0.1, do_sample=False)[0]["generated_text"]

        # Extract prediction
        predicted = extract_predicted_answer(response)

        # Logging
        print("=" * 50)
        print(f"Sample #{i+1}")
        print(f"QUESTION:\n{question}")
        print(f"GOLD ANSWER:\n{gold_answer}")
        print(f"MODEL RESPONSE:\n{response}")
        print(f"EXTRACTED PREDICTED ANSWER: {predicted}")
        print(f"CORRECT: {predicted == gold_answer}")
        print("=" * 50)

        if predicted == gold_answer:
            correct += 1

    print(f"\n[SUMMARY]")
    print(f"Total Correct: {correct}/{total}")
    print(f"Accuracy: {correct / total * 100:.2f}%")

# === Main ===
if __name__ == "__main__":
    print("Loading model...")
    pipe = load_llama_model()
    print("Loading dataset...")
    sample_data = load_sample_data(5)
    print("Running debug evaluation...")
    run_debug(pipe, sample_data)
