# 实现 y = w * x + b  可视化
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]

def forward(x):
    return x * w + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w = np.arange(0.0, 4.1, 0.1)
b = np.arange(0.0, 4.1, 0.1)
[w, b] = np.meshgrid(w, b)

l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val
    print('\t', x_val, y_val, y_pred_val, loss_val)

# draw the 3D graph
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(w, b, l_sum / 3, cmap='rainbow')
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("mse")
ax.set_title("y=w*x+b")
plt.show()