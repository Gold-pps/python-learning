class Wizard:
    def __init__(self,name):
        if not name:
            raise ValueError("Missing name")
        self.name = name

class Student(Wizard):
    def __inti__(self,name,house):
        super().__init__(name)
        self.house = house

class Professor(Wizard):
    def __init__(self,name,subject):
        super().__init__(name)
        self.subject = subject

wizarf = Wizard("E")
student = Student("A","B")
professor = Professor("C","D")
