# 使用 softmax 实现 mnist 多分类

import torch
from torchvision import transforms          # For constructing DataLoader
from torchvision import datasets            # For constructing DataLoader
from torch.utils.data import DataLoader     # For constructing DataLoader
import torch.nn.functional as F             # For using function relu()
import torch.optim as optim                 # For constructing Optimizer

# 1. Perpare datset
batch_size = 64
# 转变为像素张量 z[28*28] 0~255 -> r[1*28*28] c x w x h 0~1
# 0.1307:均值, 0.3081:标准差
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='dataset/mnist/',
                               train=True,
                               download=False,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='dataset/mnist',
                              train=False,
                              download=False,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

# 2. Design model using Class
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # 改变它的形状，变成一个矩阵
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()

# 3. Construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 4. Training cycle + Test
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299: # 每 300 轮输出一次
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0  # 正确多少
    total = 0    # 总数多少
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accurary on test set: %d %%' % (100 * correct /total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()