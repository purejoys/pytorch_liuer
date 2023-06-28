# 使用 PyTorch 实现线性回归
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
# 这样构造的 module 会自动实现 backward
    def __init__(self): # 构造函数--初始化
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x): # 前馈
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False) # 损失函数

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 优化器

# 训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    optimizer.zero_grad() # 归零
    loss.backward()
    optimizer.step() # 更新

# ouput weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# Test model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)