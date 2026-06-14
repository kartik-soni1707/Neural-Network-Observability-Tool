import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNet(nn.Module):                 
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # input layer  -> hidden 1
        self.fc2 = nn.Linear(256, 128)   # hidden 1     -> hidden 2
        self.fc3 = nn.Linear(128, 10)    # hidden 2     -> output (10 classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)        # FIX 6: flatten [batch,1,28,28] -> [batch,784]
        x = F.relu(self.fc1(x))          # layer 1, then "bend" it
        x = F.relu(self.fc2(x))          # layer 2, then "bend" it
        x = self.fc3(x)                  
        return x


EPOCHS = 5                                   

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST's known mean/std
])
train_set = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64,  shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1000, shuffle=False)

my_net = NeuralNet()

criterion = nn.CrossEntropyLoss()            
optimizer = torch.optim.SGD(my_net.parameters(), lr=0.01)

for epoch in range(EPOCHS):
    for images, labels in train_loader:   # each pass pulls one batch (your X and Y)
        optimizer.zero_grad()         # 1. clear last step's slopes
        preds = my_net(images)        # 2. forward pass: predict
        loss = criterion(preds, labels)  # 3. score the wrongness (one number)
        loss.backward()               # 4. autograd: dump slopes into every .grad
        optimizer.step()              # 5. optimizer: spend slopes, update knobs
    print(f"epoch {epoch}   loss {loss.item():.4f}")


correct = total = 0
with torch.no_grad():                        # no gradients needed when just evaluating
    for images, labels in test_loader:
        guesses = my_net(images).argmax(dim=1)   # highest-scoring digit per image
        correct += (guesses == labels).sum().item()
        total += labels.size(0)

print(f"\ntest accuracy: {100 * correct / total:.2f}%")