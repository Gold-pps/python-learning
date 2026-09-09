import re

email = input("What's your email?").strip()

if re.search(r"^\w+@(\w+\.)?\w+\.edu$",email,re.IGNORECASE):
    print("Vaild")
else:
    print("Invaild")

"""username,domain=email.split("@")

if username and domain.endswith(".edu"):
    print("Vaild")
else:
    print("Invaild")"""