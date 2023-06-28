# 导入相关库
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 先了解一下数据的基本情况，再进行我们后面的建模流程
data = pd.read_csv('otto/train.csv')
data.head()  #查看前五行
data.info()  #查看数据信息
data.describe()  #查看数据统计信息
data.isnull().sum()  #查看缺失值
data['target'].value_counts()  #查看各个类别的数量

# 1、准备数据
class OttoDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('otto/train.csv', delimiter=',', skiprows=1, usecols=np.arange(1, 94))
        df = pd.read_csv('otto/train.csv', sep=',')
        df['target'] = df['target'].map({'Class_1': 1, 'Class_2': 2,  #通过映射将类别转换为数字
                                         'Class_3': 3, 'Class_4': 4,
                                         'Class_5': 5, 'Class_6': 6,
                                         'Class_7': 7, 'Class_8': 8,
                                         'Class_9': 9})
        df['target'] = df['target'].astype('float')
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :])
        self.y_data = torch.tensor(df['target'].values)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = OttoDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


# 2、构建模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(93, 64)
        self.l2 = torch.nn.Linear(64, 32)
        self.l3 = torch.nn.Linear(32, 16)
        self.l4 = torch.nn.Linear(16, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)


model = Net()

# 3、构建损失和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)


# 4、训练模型
def train(epoch):
    model.train()  #开启训练模式
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.float(), labels.long()
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)  # 是线性层的输出值，不是概率值
        loss = criterion(outputs, labels - 1)  #因为类别是从1开始的，所以要减1
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (epoch % 5 == 0) and batch_idx % 1000 == 999:
            print('[%d, %5d] loss: %.3f' % (epoch, batch_idx + 1, running_loss / 1000))
            running_loss = 0.0


# 5、进行预测并保存结果
def test():
    model.eval()  # 开启测试模式
    xyTest = np.loadtxt('otto/test.csv', delimiter=',', skiprows=1, usecols=np.arange(1, 94))
    df1 = pd.read_csv('otto/test.csv', sep=',')
    xy_pred = torch.from_numpy(xyTest[:, :])  # 将测试集转换为tensor
    column_list = ['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5',
                   'Class_6', 'Class_7', 'Class_8', 'Class_9']
    d = pd.DataFrame(0, index=np.arange(xy_pred.shape[0]), columns=column_list)  # 创建一个空的DataFrame
    d.iloc[:, 1:] = d.iloc[:, 1:].astype('float')
    d['id'] = df1['id']  # 将id列赋值

    output = model(xy_pred.clone().detach().requires_grad_(True).float())
    row = F.softmax(output, dim=1).data  # 将输出值转换为概率值。注意维度为1
    classes = row.numpy()  # 将tensor转换为numpy
    classes = np.around(classes, decimals=2)  # 保留两位小数
    d.iloc[:, 1:] = classes  # 将概率值赋值给DataFrame
    d.to_csv('Submission.csv', index=False)  # 保存结果


if __name__ == '__main__':
    for epoch in range(1, 30):
        train(epoch)
    test()
