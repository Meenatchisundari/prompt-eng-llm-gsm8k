import torch
import dspy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

def load_qwen_dspy():
    """
    Loads Qwen/Qwen-7B-Chat model with 4-bit quantization,
    wraps it in DSPy.HFModel and returns the model.
    """
    print("[DSPy Loader] Loading Qwen-7B-Chat for DSPy...")

    model_id = "Qwen/Qwen-7B-Chat"
    hf_token = os.environ.get("HF_TOKEN", None)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        token=hf_token,
        trust_remote_code=True
    )

    dspy_model = dspy.HFModel(
        model=model,
        tokenizer=tokenizer,
        max_tokens=300,
        temperature=0.1
    )

    print("[DSPy Loader] Qwen-7B-Chat loaded and wrapped for DSPy.")
    return dspy_model
