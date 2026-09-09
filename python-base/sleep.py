def main():
    n = int(input("What's n?"))
    for i in sheep(n):
        print(i,sep="  ")

def sheep(n):
    for i in range(n):
        yield "a"*(i+1)

if __name__ == "__main__":
    main()