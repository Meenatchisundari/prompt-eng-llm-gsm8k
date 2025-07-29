# utils/dataset.py

import json
import requests

def download_gsm8k_dataset():
    """
    Download and parse the GSM8K test set from the OpenAI repository.
    Returns:
        dict: A dictionary containing the 'test' set and an empty 'train' list.
    """
    print("\nGSM8K Dataset Acquisition:")
    print("-" * 30)

    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

    try:
        print("Downloading GSM8K test set from OpenAI repository...")
        response = requests.get(test_url, timeout=30)

        if response.status_code == 200:
            print("Download successful, parsing JSONL format...")

            lines = response.text.strip().split('\n')
            test_data = []
            parsing_errors = 0

            for line in lines:
                if line.strip():
                    try:
                        item = json.loads(line)
                        test_data.append({
                            'question': item['question'],
                            'answer': item['answer']
                        })
                    except json.JSONDecodeError:
                        parsing_errors += 1
                        continue

            print(f"Successfully parsed {len(test_data)} test problems")
            if parsing_errors > 0:
                print(f"Warning: {parsing_errors} lines had parsing errors")

            return {
                'test': test_data,
                'train': []  # Placeholder for compatibility
            }

        else:
            raise requests.HTTPError(f"HTTP {response.status_code}: {response.reason}")

    except Exception as e:
        print(f"Dataset download failed: {e}")
        print("Troubleshooting tips:\n1. Check internet connection\n2. Re-run the script\n3. Verify GitHub URL")
        return None

