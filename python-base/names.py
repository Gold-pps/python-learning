"""names = []

for _ in range(3):
    names.append(input("What's your name?"))

for name in sorted(names):
    print(f"hello,{name}")"""

"""name = input("What's your name?")"""
names=[]

with open("names.txt","r") as file:
    for line in file:
        names.append(line.rstrip())

for name in sorted(names,reverse=True):
    print(f"hello,{name}")