import numpy as np
import matplotlib.pyplot as plt

# 绘制一张图
fig = plt.figure()
# 绘制3D子图
ax = fig.add_subplot(projection="3d")
x0 = np.arange(-3, 3, 0.1)
x1 = np.arange(-3, 3, 0.1)
x0, x1 = np.meshgrid(x0, x1)
z = np.array(x0 ** 2 + x1 ** 2)

ax.plot_surface(x0, x1, z, cmap='inferno')
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_zlabel("f(x)")
ax.set_title("3D")
plt.show()
