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
├── test/                     # Model tests
│   ├── llama_test.py
│   └── qwen_test.py
│
├── setup_env.py              # Hugging Face authentication setup
├── setup_path.py             # Root path setup for sys.path
├── run_all.py                # Evaluate selected strategies
├── requirements.txt          # Dependencies
└── README.md
```

## Current Capabilities

- ✅ Zero-shot prompt strategy implemented
- ✅ LLaMA-2 and Qwen model loading and testing (quantized)
- ✅ Hugging Face token setup via `setup_env.py` and Colab secrets
- ✅ Clean folder structure with modular model loaders
- ✅ Requirements file for easy setup

## Installation

```bash
!git clone https://github.com/Meenatchisundari/prompt-eng-llm-gsm8k.git
%cd prompt-eng-llm-gsm8k
!pip install -r requirements.txt
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

## Path Setup

To ensure internal modules import correctly, run this at the top of your notebook or script:

```python
exec(open("setup_path.py").read())
```

## Model Setup and Testing (LLaMA-2)

To test LLaMA-2-7B-chat-hf using quantization:

```bash
!python setup_path.py
!python setup_env.py
```

Then in Python:

```python
from models.llama2_loader import load_llama2_quantized
llama_generator = load_llama2_quantized()
```

Or run the test file:

```bash
!python test/llama_test.py
```

## Qwen Model Setup and Testing

To test Qwen2.5-7B-chat using quantization:

```bash
!python setup_path.py
!python setup_env.py
```

Then in Python:

```python
from models.qwen_loader import load_qwen_quantized
qwen_generator = load_qwen_quantized()
```

Or run the test file:

```bash
!python test/qwen_test.py
```

## Unified Strategy Evaluation (run_all.py)

To evaluate zero-shot prompts using either model:

```bash
!python3 run_all.py llama   # for LLaMA-2
!python3 run_all.py qwen    # for Qwen2.5
```

> The script will load the model, run zero-shot inference on GSM8K samples, and print predictions.

## Next Steps

- [ ] Integrate GPT-4 strategy
- [ ] Add Chain-of-Thought, Few-shot, and Prolog prompts
- [x] Evaluate and compare all strategies via `run_all.py`

---

**Author**: Meenatchi Sundari  
**License**: MIT
