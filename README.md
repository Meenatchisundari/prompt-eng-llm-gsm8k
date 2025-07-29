# Prompt Engineering on LLMs using GSM8K

This project investigates various prompt engineering strategies applied to Large Language Models (LLMs) such as GPT-4, LLaMA-2, and Qwen2.5, using the GSM8K dataset for grade school math reasoning.

## Project Overview

- **Goal**: Evaluate and compare different prompt strategies across multiple models on math reasoning tasks.
- **Dataset**: [GSM8K](https://huggingface.co/datasets/gsm8k)
- **Models Supported**:
  - GPT-4 (via OpenAI API)
  - LLaMA-2-7B-chat (via Hugging Face, quantized)
  - Qwen2.5-Math-7B (via Hugging Face, quantized)

## Project Structure

```
prompt-eng-llm-gsm8k/
├── models/                    # Model loaders
│   ├── gpt4_loader.py
│   ├── llama2_loader.py
│   └── qwen_loader.py
│
├── prompts/                  # Prompting strategies
│   ├── zero_shot.py
│   ├── cot.py
│   ├── few_shot.py
│   ├── self_consistency.py
│   └── prolog_style.py
│
├── utils/                    # Evaluation and helpers
│   ├── extraction.py
│   ├── dataset.py
│   └── evaluation.py
│
├── test_model.py             # Basic LLaMA-2 functionality test
├── setup_env.py              # Hugging Face authentication setup
├── run_all.py                # Evaluate selected strategies
├── requirements.txt          # Dependencies
└── README.md
```

## Current Capabilities

-  Zero-shot prompt strategy implemented
-  LLaMA-2 quantized loading and testing
-  Hugging Face token setup via `setup_env.py` and Colab secrets
-  Clean folder structure with modular model loaders
-  Requirements file for easy setup

## Model Setup and Testing (LLaMA-2)

To test LLaMA-2-7B-chat-hf using quantization:

```bash
pip install -r requirements.txt
python setup_env.py
```

Then in Python:

```python
from models.llama2_loader import load_llama2_quantized
llama_generator = load_llama2_quantized()
```

To run a basic math prompt test:

```bash
python test_model.py
```

## Hugging Face Authentication

Authentication is handled via `setup_env.py`:
- Tries to use the `HF_TOKEN` environment variable
- Falls back to `google.colab.userdata` if available
- Prompts for manual token input otherwise

This makes the project usable in:
- Google Colab
- Local Jupyter/VS Code
- Terminal/CLI environments

## Installation

```bash
git clone https://github.com/Meenatchisundari/prompt-eng-llm-gsm8k.git
cd prompt-eng-llm-gsm8k
pip install -r requirements.txt
```

## Next Steps

- [ ] Add Qwen model loader and test
- [ ] Integrate GPT-4 strategy
- [ ] Add Chain-of-Thought, Few-shot, and Prolog prompts
- [ ] Evaluate and compare all strategies via `run_all.py`

---

**Author**: Meenatchi Sundari  
**License**: MIT

