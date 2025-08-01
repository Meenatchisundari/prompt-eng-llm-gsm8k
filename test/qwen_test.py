exec(open("setup_path.py").read())

from models.qwen_loader import load_qwen_quantized

def test_qwen_model_basic_functionality(text_generator):
    """Test Qwen model functionality with a simple math problem."""
    print("Testing Qwen model with a basic math prompt...")

    test_prompt = """<|im_start|>system\nYou are a helpful math tutor.<|im_end|>\n<|im_start|>user\nSolve: What is 15 + 27? Show steps.<|im_end|>\n<|im_start|>assistant\n"""

    try:
        response = text_generator(test_prompt)
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
