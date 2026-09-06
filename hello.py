def main():
    name = input("what's your name? ")
    print(hello(name))

def hello(to="Gold_pps"):
    return f"hello {to}"

if __name__=="__main__":
    main()