# 实现梯度下降算法
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x): # 前馈
    return x * w

def cost(xs, ys): # 损失函数
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y) # gradient update 更新公式
    return grad / len(xs)

epcoh_list = []
cost_list = []

print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val  # 权重减去学习率乘以梯度
    print('Epoch:', epoch, 'w=', w, "cost=", cost_val)
    epcoh_list.append(epoch)
    cost_list.append(cost_val)
print('Predict (after training)', 4, forward(4))

# 绘图
plt.plot(epcoh_list, cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()