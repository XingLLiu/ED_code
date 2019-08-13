class Person:
    def __init__(self, name, age, maths):
        self.name = name
        self.age = age
        self.maths = maths
    def greeting(self):
        print("Hello, my name is {}.".format(self.name))
    def favorite_maths(self):
        print("My favorite maths is {}.".format(self.maths))


