# 实现反向传播 y = w * x
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0]) # 权重只有一个值
w.requires_grad = True  # 需要计算梯度

def forword(x):
    return x * w

# 在构建计算图
def loss(x,y):
    y_pred = forword(x)
    return (y_pred - y) ** 2

# 训练过程
print("predict (before training)", 4, forword(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward() # 梯度存在变量中，计算图会自动释放
        print('\tgrad:', x, y, w.grad.item())  # item 把数值拿出来建立标量
        w.data = w.data - 0.01 * w.grad.data # 取 data 不会建立计算图 纯数值的修改

        w.grad.data.zero_() # w 中的梯度全部清零

    print("progress:", epoch, l.item())

print("predict (after training)", 4, forword(4).item())