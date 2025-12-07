from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import combinations
from matplotlib.patches import Patch  # 用于图例

# --- 1. 数据准备 (所有特征, 三分类) ---
iris = load_iris()
X = iris.data  # 所有四个特征
y = iris.target
feature_names = [name.replace(' (cm)', '') for name in iris.feature_names]
class_names = iris.target_names

# --- 2. 准备绘图设置 ---
# 定义颜色映射: Setosa, Versicolor, Virginica
# 浅色用于绘制区域 (决策边界)
cmap_light = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])  # 浅红, 浅绿, 浅蓝
# 深色用于绘制数据点
cmap_bold = ListedColormap(['#FF0000', '#00AA00', '#0000FF'])  # 亮红, 亮绿, 亮蓝

# 训练模型 (使用 Logistic Regression)
model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200)

# --- 3. 遍历所有特征对 (共6对) ---
n_features = X.shape[1]
# combinations 生成所有两两组合的索引 [(0, 1), (0, 2), ..., (2, 3)]
feature_pairs = list(combinations(range(n_features), 2))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()  # 展平子图数组，方便遍历

for i, (f1_idx, f2_idx) in enumerate(feature_pairs):
    ax = axes[i]

    # 提取当前特征对的数据
    X_pair = X[:, [f1_idx, f2_idx]]

    # 对当前特征子集重新训练模型
    model.fit(X_pair, y)

    # 创建网格 (用于预测决策边界)
    x_min, x_max = X_pair[:, 0].min() - .5, X_pair[:, 0].max() + .5
    y_min, y_max = X_pair[:, 1].min() - .5, X_pair[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策区域 (着色区域)
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

    # 绘制数据点
    ax.scatter(X_pair[:, 0], X_pair[:, 1], c=y, cmap=cmap_bold,
               edgecolor='k', s=20)

    # 设置标签和标题
    f1_name = feature_names[f1_idx]
    f2_name = feature_names[f2_idx]
    ax.set_xlabel(f1_name)
    ax.set_ylabel(f2_name)
    ax.set_title(f'Decision Boundary: {f1_name} vs {f2_name}')

# 调整布局，避免重叠
plt.tight_layout()

# 添加统一图例
legend_elements = [Patch(facecolor=cmap_bold.colors[i], edgecolor='k', label=class_names[i])
                   for i in range(len(class_names))]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, title="Classes")

plt.show()