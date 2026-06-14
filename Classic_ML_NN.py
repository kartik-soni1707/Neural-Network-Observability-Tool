import torch
import torch.nn as nn
import torch.nn.functional as F

#Tensors
a = torch.tensor([1.0, 2.0, 3.0])      # straight from a Python list
b = torch.zeros(2, 3)                   # a 2x3 grid of zeros
c = torch.ones(2, 3)                    # a 2x3 grid of ones
d = torch.rand(2, 3)                    # 2x3 of random numbers in [0, 1)

print(a.shape)
print(a.dtype)
print(c*d)

#Concept of autograd
w = torch.tensor(4.0, requires_grad=True)   # a knob we want to tune
x = torch.tensor(3.0)                        # fixed input, no requires_grad

y = w * x                                    # PyTorch records: "y came from w times x"
y.backward()                                 # how does y change as w changes?

print(w.grad)    # tensor(3.)  -> because y = w*x, so dy/dw = x = 3
print(x.grad)    # None        -> x wasn't a knob, so nothing was tracked for it
class BiggerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # input layer  -> hidden 1
        self.fc2 = nn.Linear(256, 128)   # hidden 1     -> hidden 2
        self.fc3 = nn.Linear(128, 10)    # hidden 2     -> output (10 classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))          # layer 1, then "bend" it
        x = F.relu(self.fc2(x))          # layer 2, then "bend" it
        x = F.softmax(self.fc3(x))                  # final scores (no bend on the last one)
        return x

model = BiggerNet()
X = torch.rand(784)
print(model(X))