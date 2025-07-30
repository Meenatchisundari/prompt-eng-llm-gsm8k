FEW_SHOT_EXAMPLES = '''
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: I need to find how many trees were planted. I start with 15 trees and end with 21 trees. So 21 - 15 = 6 trees were planted. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: I start with 3 cars. Then 2 more cars arrive. So I have 3 + 2 = 5 cars total. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: First, I find the total chocolates: 32 + 42 = 74 chocolates. Then they ate 35, so they have 74 - 35 = 39 chocolates left. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops and ended with 12 lollipops. So he gave away 20 - 12 = 8 lollipops to Denny. The answer is 8.
'''

def few_shot_prompt(question: str) -> str:
    return f"{FEW_SHOT_EXAMPLES.strip()}\n\nQ: {question}\nA:"
