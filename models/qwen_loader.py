import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

def load_qwen_quantized():
    """Load Qwen-7B-Chat model with 4-bit quantization."""
    print("Loading Qwen-7B-Chat model with quantization...")

    model_id = "Qwen/Qwen-7B-Chat"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # ---- Patch to fix 'past_key_values' crash in Qwen ----
    original_forward = model.forward

    def patched_forward(*args, **kwargs):
        if 'past_key_values' in kwargs and kwargs['past_key_values'] is None:
            kwargs.pop('past_key_values')
        return original_forward(*args, **kwargs)

    model.forward = patched_forward
    # ------------------------------------------------------


    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    print("Qwen model loaded successfully.")
    return generator

