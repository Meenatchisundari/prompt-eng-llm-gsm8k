import os
import torch
import dspy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_llama2_dspy():
    """
    Loads LLaMA-2-7B-chat-hf with 4-bit quantization and wraps it in a DSPy-compatible model.
    """
    print("[DSPy Loader] Loading LLaMA-2-7B-chat-hf for DSPy...")

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        token=hf_token
    )

    class LocalHFWrapper(dspy.BaseLM):
        def __init__(self):
            super().__init__(model=model)
            self.model = model
            self.tokenizer = tokenizer
            self.max_tokens = 300
            self.temperature = 0.1

        def forward(self, prompt, **kwargs):
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model.generate(
                input_ids,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    return LocalHFWrapper()
