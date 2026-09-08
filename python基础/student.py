class Student:
    def __init__(self,name,house):
        self.name = name
        self.house = house

    def __str__(self):
        return f"{self.name} from {self.house}"

    """def charm(self):
        match self.patronus:
            case "a":
                return "😊"
            case "b":
                return "😂"
            case "c":
                return "🤣"
            case _:
                return "❤️" """

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self,name):
        if not name:
            raise ValueError("Missing name")
        self._name = name

    @property
    def house(self):
        return self._house

    @house.setter
    def house(self,house):
        if house not in ["A","B","C"]:
            raise ValueError("Invalid house")
        self._house = house

    @classmethod
    def get(cls):
        name = input("Name: ")
        house = input("House: ")
        return cls(name,house)


def main():
    student = Student.get()
    print(student)

if __name__=="__main__":
    main()