import random

class Hat:
    houses=["A","B","C"]

    @classmethod    
    def sort(cls,name):
        print(name, "is in",random.choice(cls.houses))

Hat.sort("Harry")