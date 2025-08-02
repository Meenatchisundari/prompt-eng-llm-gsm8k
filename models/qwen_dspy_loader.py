import os
import torch
import dspy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_qwen_dspy():
    """
    Loads Qwen/Qwen-7B-Chat model with 4-bit quantization and wraps it in a DSPy-compatible model.
    """
    print("[DSPy Loader] Loading Qwen-7B-Chat for DSPy...")

    model_id = "Qwen/Qwen-7B-Chat"
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config,
        token=hf_token
    )

    class QwenDSPyLM(dspy.BaseLM):
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

    return QwenDSPyLM()
