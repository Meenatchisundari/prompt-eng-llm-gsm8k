# setup_path.py
import os
import sys

if "__file__" in globals():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    root_path = os.path.abspath("/content/prompt-eng-llm-gsm8k")

if root_path not in sys.path:
    sys.path.append(root_path)
    print(f"[setup_path] Root path added: {root_path}")
else:
    print(f"[setup_path] Root path already exists: {root_path}")
