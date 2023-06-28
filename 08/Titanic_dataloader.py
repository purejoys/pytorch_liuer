# 实现 titanic 分类器

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1、加载数据集
class TitanicDataset(Dataset):
    def __init__(self, filepath):
        features = ['Pclass','Sex','SibSp','Parch','Fare']
        data = pd.read_csv(filepath)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(np.array(pd.get_dummies(data[features])).astype(np.float32))
        self.y_data = torch.from_numpy(np.array(data['Survived']).astype(np.float32))


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = TitanicDataset('titanic/train.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

# 2、设计模型
class TitanicNet(torch.nn.Module):
    def __init__(self):
        super(TitanicNet, self).__init__()
        self.l1 = torch.nn.Linear(6, 4)
        self.l2 = torch.nn.Linear(4, 2)
        self.l3 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.l1(x))
        x = self.sigmoid(self.l2(x))
        x = self.sigmoid(self.l3(x))
        return x

model = TitanicNet()

# 3、设计优化器和损失函数
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_list = []

# 4、训练测试
for epoch in range(100):
    sum = 0.0
    for i, data in enumerate(train_loader, 0):
        # prepare data
        inputs, labels = data
        # forward
        y_pred = model(inputs)
        y_pred = y_pred.squeeze(-1)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        sum += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()

    loss_list.append(sum / train_loader.batch_size)

# 可视化
num_list = [i for i in range(len(loss_list))]
plt.plot(num_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()