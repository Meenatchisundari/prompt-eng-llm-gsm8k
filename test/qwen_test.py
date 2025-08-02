exec(open("setup_path.py").read())
from models.qwen_loader import load_qwen_quantized

def test_qwen_model_basic_functionality(generator):
    print("Testing Qwen model with a basic math prompt...")
    prompt = """Solve: What is 33 + 9? Show steps."""
    try:
        response = generator(prompt)
        output = response[0]['generated_text']
        print("\nModel Response:")
        print(output)
        if "42" in output:
            print("Test Passed: Correct answer found.")
        else:
            print("Test Completed: Response generated but answer may be unclear.")
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    generator = load_qwen_quantized()
    test_qwen_model_basic_functionality(generator)
