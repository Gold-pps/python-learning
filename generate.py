import random
import statistics
import sys


if len(sys.argv)<2:
    sys.exit("A")

for i in sys.argv[1:]:
    print("hello,my name is",i)

"""print(statistics.mean([2,3]))
"""
"""cards = ["A","B","C","D","E"]
random.shuffle(cards)
for card in cards:
    print(card)"""