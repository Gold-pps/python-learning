"""students = [
    {"name":"Hermione","house":"Gryffindor"},
    {"name":"Draco","house":"Slytherin"},
    {"name":"Harry","house":"Gryffindor"},
    {"name":"Ron","house":"Gryffindor"},
    {"name":"Padma","house":"Ravenclaw"}
]

def is_gryffindor(s):
    return s["house"] == "Gryffindor"

gryffindors = filter(is_gryffindor,students)

for gryffindor in sorted(gryffindors,key=lambda s:s["name"]):
    print(gryffindor["name"])"""

students = ["a","b","c"]

g = {student: "g" for student in students}

print(g)

for i,student in enumerate(students):
    print(i+1,student)