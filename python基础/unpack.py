"""def total(g,s,k):
    return (g*17+s)*29+k

print(total(100,50,25))

coins = {"g":100,"s":50,"k":25}

print(total(**coins))"""
def f(*args,**kwargs):
    print("Positional:",args)

f(100,50,25)