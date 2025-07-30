# Manually execute setup_path.py so sys.path is updated in this process
exec(open("setup_path.py").read())

from models.qwen_loader import load_qwen_quantized

def test_qwen():
    generator = load_qwen_quantized()
    prompt = "What is the sum of 128 and 349?"
    response = generator(prompt, max_new_tokens=100, temperature=0.1)[0]["generated_text"]
    print("Prompt:", prompt)
    print("Response:", response)

if __name__ == "__main__":
    test_qwen()
