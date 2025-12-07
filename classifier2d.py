from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- 1. 数据准备 ---
iris = load_iris()
# 仅选择后两个特征 (Petal Length and Petal Width)
X = iris.data[:, 2:]
y = iris.target
feature_names = iris.feature_names[2:]
target_names = iris.target_names

# 划分数据集 (可以省略测试集划分，因为这里只关注可视化所有数据点的决策边界)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建绘图网格
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# --- 2. 定义分类器字典 ---
classifiers = {
    "Logistic Regression (C=0.1)": LogisticRegression(C=0.1, max_iter=200, random_state=42),
    "Logistic Regression (C=1)": LogisticRegression(C=1, max_iter=200, random_state=42),
    "KNeighbors (n=5)": KNeighborsClassifier(5),
    "Decision Tree (Depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    # Kernel SVC 的 predict_proba 相对较慢
    "Linear SVM": SVC(kernel="linear", C=0.5, probability=True, random_state=42)
}

# 绘图颜色设置
# 用于决策边界/概率图的颜色
cmap_boundary = mcolors.ListedColormap(['#FFC600', '#56C42D', '#3266FF'])  # 黄、绿、蓝
# 用于散点的颜色
cmap_points = mcolors.ListedColormap(['yellow', 'green', 'blue'])
n_classifiers = len(classifiers)

# 创建图形，每行一个分类器，每行有 4 列 (3 个概率图 + 1 个最大类别图)
fig, axs = plt.subplots(n_classifiers, 4, figsize=(16, 4 * n_classifiers))
plt.subplots_adjust(wspace=0.1, hspace=0.3)

# --- 3. 迭代训练和可视化 ---
for i, (name, classifier) in enumerate(classifiers.items()):

    # 训练模型
    classifier.fit(X_train, y_train)

    # 获取网格点的概率
    Z_prob = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z_prob = Z_prob.reshape(xx.shape[0], xx.shape[1], 3)

    # 获取网格点的最大类别 (决策边界)
    Z_max_class = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # 绘制 4 个子图: Class 0 Probability, Class 1 Probability, Class 2 Probability, Max Class
    for j in range(4):
        ax = axs[i, j]
        ax.set_xticks(())  # 隐藏x轴刻度
        ax.set_yticks(())  # 隐藏y轴刻度

        # 绘制概率图 (j=0, 1, 2)
        if j < 3:
            class_index = j
            prob_map = Z_prob[:, :, class_index]

            # 使用白色到类别颜色渐变
            class_color_hex = mcolors.to_hex(cmap_boundary(class_index))
            cmap_prob = mcolors.LinearSegmentedColormap.from_list(
                f'class_{class_index}_prob_cmap', ['white', class_color_hex], N=256)

            ax.contourf(xx, yy, prob_map, levels=np.linspace(0, 1, 21), alpha=0.8, cmap=cmap_prob)
            ax.set_title(f'Class {class_index}', size=10)

        # 绘制最大类别图 (j=3)
        else:
            ax.contourf(xx, yy, Z_max_class, alpha=0.8, cmap=cmap_boundary)
            ax.set_title('Max class', size=10)

        # 在每个图上绘制原始数据点
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=20, cmap=cmap_points)

        # 在每行第一列添加分类器名称
        if j == 0:
            ax.set_ylabel(name, rotation=90, size=10, labelpad=10)

# 在最底部添加图例
cbar_ax_prob = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # 调整位置 [left, bottom, width, height]
# 创建一个通用的概率颜色条 (例如 Class 2 的蓝色渐变)
cmap_prob_general = mcolors.LinearSegmentedColormap.from_list(
    'prob_general_cmap', ['white', 'blue'], N=256)
sm_prob = plt.cm.ScalarMappable(cmap=cmap_prob_general)
sm_prob.set_array(np.linspace(0, 1, 100))
cbar_prob = fig.colorbar(sm_prob, cax=cbar_ax_prob, orientation="horizontal", ticks=np.linspace(0, 1, 6))
cbar_prob.set_label("Probability")

# 在右下角添加 Max Class 图例
# 因为 Max Class 是离散的，我们手动创建图例
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=cmap_boundary(0), label=f'Probability class 0 ({target_names[0]})'),
    Patch(facecolor=cmap_boundary(1), label=f'Probability class 1 ({target_names[1]})'),
    Patch(facecolor=cmap_boundary(2), label=f'Probability class 2 ({target_names[2]})')
]
# 添加到图外侧
axs[-1, -1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.05), title="Max Class Legend")

plt.show()