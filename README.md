# 📚 GSM8K Prompt Engineering Evaluation

This project benchmarks LLaMA-2-7B-Chat and Qwen-7B-Chat models on the GSM8K dataset using different prompt engineering strategies. It supports 4-bit quantized inference for LLaMA-2 via `bitsandbytes`, and float16 inference for Qwen.

---

## 📦 Features

* ✅ Zero-shot, Chain-of-Thought, Few-shot, Prolog-style, and Self-Consistency prompt strategies
* ✅ LLaMA-2 with 4-bit quantization using Hugging Face + `bitsandbytes`
* ✅ Qwen with float16 and safe tokenizer support
* ✅ GSM8K test set auto-downloaded and parsed
* ✅ Evaluation logs + CSV result output

---

## 📁 Directory Structure

```
├── models/
│   ├── llama2_loader.py         # Loads quantized LLaMA-2 model
│   └── qwen_loader.py           # Loads float16 Qwen model
│
├── prompts/                    # Prompting strategies
│   ├── zero_shot.py             # Zero-shot prompting logic
│   ├── cot.py                   # Chain-of-thought prompting logic
│   ├── few_shot.py              # Few-shot examples logic
│   ├── self_consistency.py      # Self-consistency ensemble logic
│   └── prolog_style.py          # Prolog-style formal prompting
│
├── utils/
│   ├── dataset.py              # Loads GSM8K dataset
│   ├── evaluation.py           # Evaluation loop and logging
│   ├── extraction.py           # Answer number extraction via regex
│
├── run_all.py                  # Runs full evaluation for a given model + N samples
├── llama_test.py               # Quick test script for LLaMA-2
├── qwen_test.py                # Quick test script for Qwen
├── requirements.txt            # All required libraries
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-org/gsm8k-eval-pipeline.git
cd gsm8k-eval-pipeline
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run LLaMA or Qwen Evaluation

```bash
python run_all.py llama 20   # for LLaMA-2 (4-bit)
python run_all.py qwen  20   # for Qwen (float16)
```

---

## 🧪 Test Scripts

```bash
python llama_test.py    # Check LLaMA inference sanity
python qwen_test.py     # Check Qwen output
```

---

## 🧠 Requirements Summary

See `requirements.txt` — includes:

* transformers, torch, accelerate, bitsandbytes
* pandas, numpy, requests, datasets
* openai (optional)
* sentencepiece, protobuf (for tokenizer safety)

---

## 📝 Output Files

* `results/results_<model>_<n>_<timestamp>.csv` – per-strategy results
* `results/<model>_incorrect_<strategy>.txt` – error logs

---

## 📌 License

MIT License

---

## 🤝 Contributors

* Meenatchi Sundari Muthirulappan (Author)
  
---

## 📬 Contact

For questions or extensions, open an issue or reach out via GitHub Discussions.
