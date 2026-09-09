name = input("What's your name?")

match name:
    case "A" | "B":
        print("ND")
    case _:
        print("Who?")