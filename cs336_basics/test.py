class A():
    def __init__(self):
        self.a = 10
        self.b = 20
    def change(self): 
        a = self.a
        b = self.b

        a = 20
        print(a, self.a)

        b = 30
        print(b, self.b)

A = A()
A.change()