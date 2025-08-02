import json
import requests

def download_gsm8k_dataset():
    print("\nGSM8K Dataset Acquisition:\n" + "-" * 30)

    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    try:
        print("Downloading GSM8K test set from OpenAI repository...")
        response = requests.get(test_url, timeout=30)
        if response.status_code == 200:
            print("Download successful, parsing JSONL format...")
            lines = response.text.strip().split('\n')
            test_data = []
            for line in lines:
                if line.strip():
                    try:
                        item = json.loads(line)
                        test_data.append({
                            'question': item['question'],
                            'answer': item['answer']
                        })
                    except json.JSONDecodeError:
                        continue
            print(f"Successfully parsed {len(test_data)} test problems")
            return {'test': test_data, 'train': []}
        else:
            raise requests.HTTPError(f"HTTP {response.status_code}: {response.reason}")
    except Exception as e:
        print(f"Dataset download failed: {e}")
        return None


