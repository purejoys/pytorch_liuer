# 使用 softmax 实现 otto 数据集的分类

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# 1、加载数据集
class OttoDataset(Dataset):
    def __init__(self, filepath):
        features = ['feat_1','feat_2','feat_3','feat_4','feat_5',
                    'feat_6','feat_7','feat_8','feat_9','feat_10','feat_11',
                    'feat_12','feat_13','feat_14','feat_15','feat_16',
                    'feat_17','feat_18','feat_19','feat_20','feat_21','feat_22',
                    'feat_23','feat_24','feat_25','feat_26','feat_27','feat_28',
                    'feat_29','feat_30','feat_31','feat_32','feat_33','feat_34',
                    'feat_35','feat_36','feat_37','feat_38','feat_39','feat_40',
                    'feat_41','feat_42','feat_43','feat_44','feat_45','feat_46',
                    'feat_47','feat_48','feat_49','feat_50','feat_51','feat_52',
                    'feat_53','feat_54','feat_55','feat_56','feat_57','feat_58',
                    'feat_59','feat_60','feat_61','feat_62','feat_63','feat_64',
                    'feat_65','feat_66','feat_67','feat_68','feat_69','feat_70',
                    'feat_71','feat_72','feat_73','feat_74','feat_75','feat_76',
                    'feat_77','feat_78','feat_79','feat_80','feat_81','feat_82',
                    'feat_83','feat_84','feat_85','feat_86','feat_87','feat_88',
                    'feat_89','feat_90','feat_91','feat_92','feat_93']
        data = pd.read_csv(filepath)
        data["target"] = data["target"].map({'Class_1':1,'Class_2':2,'Class_3':3,
                                             'Class_4':4,'Class_5':5,'Class_6':6,
                                             'Class_7':7,'Class_8':8,'Class_9':9})
        data['target'] = data['target'].astype('float64')
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(np.array(data[features]).astype(np.float32))
        self.y_data = torch.tensor(data['target'].values)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = OttoDataset('otto/train.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

# 2、设计模型
class OttoModel(torch.nn.Module):
    def __init__(self):
        super(OttoModel, self).__init__()
        self.l1 = torch.nn.Linear(93, 86)
        self.l2 = torch.nn.Linear(86, 74)
        self.l3 = torch.nn.Linear(74, 64)
        self.l4 = torch.nn.Linear(64, 32)
        self.l5 = torch.nn.Linear(32, 16)
        self.l6 = torch.nn.Linear(16, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.l6(x)

model = OttoModel()

# 3、设计优化器和损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 4、训练和测试
def train(epoch):
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader, 0):
        inputs, target = inputs.float(), target.long()
        optimizer.zero_grad()

        y_pred = model(inputs)
        y_pred = y_pred.squeeze(-1)
        loss = criterion(y_pred, target-1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)