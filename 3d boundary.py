from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # 引入标准化工具
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.patches import Patch # 导入 Patch 修复 NameError

# --- 1. 数据准备 (三特征，两分类) ---
iris = load_iris()

# 新的选择：前三个特征 (Sepal Length, Sepal Width, Petal Length)
X_full = iris.data[:, :3]
y_full = iris.target
feature_names = iris.feature_names[1:]
target_names = iris.target_names

# 为了实现两分类，我们只保留类别 1 (Versicolor) 和 类别 2 (Virginica)
X = X_full[y_full >= 1]
y = y_full[y_full >= 1] - 1  # 重新编码: 1 -> 0, 2 -> 1

# --- 2. 特征标准化 (使数据分布在相似的范围内，如 PPT 所示) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # 对数据进行标准化

# --- 3. 训练逻辑回归模型 ---
# 逻辑回归在二分类中会找到一个线性超平面
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_scaled, y) # 使用标准化后的数据 X_scaled 进行训练

# 模型的系数和截距定义了决策超平面: W0*x1 + W1*x2 + W2*x3 + b = 0
W = model.coef_[0]
b = model.intercept_[0]
print(f"模型系数 W (x1, x2, x3): {W}")
print(f"模型截距 b: {b}")

# --- 4. 3D 可视化 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点
# 类别 0 (Versicolor) -> 蓝, 类别 1 (Virginica) -> 红
cmap_points = mcolors.ListedColormap(['blue', 'red'])
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
           c=y, cmap=cmap_points, s=50, edgecolors='k') # 使用 X_scaled 绘图

# --- 绘制决策超平面 ---

# 定义 x1 和 x2 的范围 (使用标准化后的数据范围)
x1_min, x1_max = X_scaled[:, 0].min(), X_scaled[:, 0].max()
x2_min, x2_max = X_scaled[:, 1].min(), X_scaled[:, 1].max()

# 创建网格点 for x1, x2
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 20),
                       np.linspace(x2_min, x2_max, 20))

# 计算超平面上对应的 x3 值： x3 = -(W0*x1 + W1*x2 + b) / W2
xx3 = (-W[0] * xx1 - W[1] * xx2 - b) / W[2]

# 绘制超平面
ax.plot_surface(xx1, xx2, xx3, alpha=0.5, color='gray', antialiased=False)

# 设置标签和标题 (使用简化的缩放标签)
ax.set_xlabel('X1 (Sepal Width, Scaled)')
ax.set_ylabel('X2 (Petal Length, Scaled)')
ax.set_zlabel('X3 (Petal Width, Scaled)')
ax.set_title('3D Decision Hyperplane for Binary Classification')

# 调整轴限制，使其更接近 PPT 中 -2.0 到 2.0 的范围 (可选)
# ax.set_xlim([-2.5, 2.5])
# ax.set_ylim([-2.5, 2.5])
# ax.set_zlim([-2.5, 2.5])

# 调整视角以便更好地观察超平面
ax.view_init(elev=20, azim=135)

# 添加图例
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label=target_names[1], markerfacecolor='blue', markersize=10, markeredgecolor='k'),
    plt.Line2D([0], [0], marker='o', color='w', label=target_names[2], markerfacecolor='red', markersize=10, markeredgecolor='k'),
    Patch(facecolor='gray', alpha=0.5, label='Decision Hyperplane') # 使用 Patch 绘制超平面的图例
]
ax.legend(handles=legend_elements, loc='best')

plt.show()