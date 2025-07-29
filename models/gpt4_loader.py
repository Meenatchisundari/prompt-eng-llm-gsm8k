import openai

def query_gpt4(prompt):
    """Query the GPT-4 model via OpenAI API"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"GPT-4 API request failed: {e}")
        return None
