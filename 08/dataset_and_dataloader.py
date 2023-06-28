# 加载数据集

import numpy as np
import torch
from torch.utils.data import Dataset  # DataLoader 抽象类
from torch.utils.data import DataLoader

# 1. Prepare datset
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index): # 能够支持下标操作
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 数据集的条数
        return self.len

dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2) # num_worker 并行的进程为2

# 2. Design model using Class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

# 3. Construct loss and optimizer
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. Training cycle
for epoch in range(100):
    for i, data in enumerate(train_loader, 0): # mini-batch
        # 1. Prepare data
        inputs, lables = data
        # 2. Forward
        y_pred = model(inputs)
        loss = criterion(y_pred, lables)
        print(epoch, i, loss.item())
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        # 4. Update
        optimizer.step()