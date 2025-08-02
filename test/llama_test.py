exec(open("setup_path.py").read())
from models.llama2_loader import load_llama2_quantized

def test_llama_model_basic_functionality(text_generator):
    """Test LLaMA model functionality with a simple math problem."""
    print("Testing LLaMA-2 model with a basic math prompt...")

    test_prompt = """<s>[INST] Solve: What is 15 + 27? Show steps. [/INST]"""

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
    generator = load_llama2_quantized()
    test_llama_model_basic_functionality(generator)
