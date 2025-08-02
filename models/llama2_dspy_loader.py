import torch
import dspy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os

def load_llama2_dspy():
    """
    Loads the LLaMA-2-7B-chat-hf model with 4-bit quantization,
    wraps it in DSPy.HFModel and returns the model object.
    """
    print("[DSPy Loader] Loading LLaMA-2-7B-chat-hf for DSPy...")

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    hf_token = os.environ.get("HF_TOKEN", None)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        token=hf_token
    )

    dspy_model = dspy.HFModel(
        model=model,
        tokenizer=tokenizer,
        max_tokens=300,
        temperature=0.1
    )

    print("[DSPy Loader] LLaMA-2 model wrapped for DSPy.")
    return dspy_model
