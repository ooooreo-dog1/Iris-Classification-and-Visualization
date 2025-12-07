from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objects as go

# --- 1. 数据准备 (三特征，两分类) ---
iris = load_iris()
# 使用三个特征进行训练
X_full = iris.data[:, :3]
y_full = iris.target

# 仅保留两分类
X = X_full[y_full >= 1]
y = y_full[y_full >= 1] - 1
feature_names = iris.feature_names[:3]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. 训练非线性模型 (3个特征) ---
# 使用高斯过程，产生类似图片中的平滑波浪形状
kernel = 1.0 * RBF(length_scale=1.0)
model = GaussianProcessClassifier(kernel=kernel, random_state=42)
model.fit(X_scaled, y)

# --- 3. 生成曲面数据 (切片法) ---
# 为了画出图片中的 "地形图" (Surface)，我们必须固定一个维度。
# 这里我们固定 X3 (Petal Length) 为其平均值，只展示 X1 和 X2 的变化对概率的影响。

# 定义 X1, X2 的网格
x_range = np.linspace(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 50)
y_range = np.linspace(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1, 50)
xx, yy = np.meshgrid(x_range, y_range)

# 固定 X3 为 0 (因为数据已标准化，0 代表平均值)
zz_fixed = np.zeros_like(xx.ravel())

# 组合成 (N, 3) 的输入矩阵，满足模型对3个特征的需求
X_grid = np.c_[xx.ravel(), yy.ravel(), zz_fixed]

# 预测概率 P(Class 1)
# 结果会形成山峰 (Class 1, Red) 和山谷 (Class 0, Blue)
probs = model.predict_proba(X_grid)[:, 1]
Z = probs.reshape(xx.shape)

# --- 4. Plotly 可视化 (完全复刻图片风格) ---
fig = go.Figure()

# 绘制曲面
fig.add_trace(go.Surface(
    z=Z, x=xx, y=yy,
    # 颜色映射：类似图片中的 蓝 -> 白 -> 红
    colorscale='RdBu_r',
    cmin=0, cmax=1,
    opacity=0.9,
    name='Probability Surface',

    # 关键设置：添加底部投影和网格线
    contours={
        "z": {
            "show": True,  # 显示等高线
            "start": 0, "end": 1,  # 范围
            "size": 0.05,  # 密度
            "color": "white",  # 网格线颜色 (类似图片中的网格)
            "project": {"z": True}  # 开启底部投影 (重点！)
        }
    }
))

# 绘制原始数据点 (投影到曲面上方或下方，可选)
# 为了让图跟PPT一样干净，这里可以选择不画点，或者只画少许

# 布局设置
fig.update_layout(
    title='Task 3: 3D Probability Surface (Sliced at X3=Mean)',
    scene=dict(
        xaxis_title=f'{feature_names[0]} (X1)',
        yaxis_title=f'{feature_names[1]} (X2)',
        zaxis_title='Probability P(Class 1)',

        # 限制 Z 轴范围，留出底部投影的空间
        zaxis=dict(range=[-0.1, 1.1]),

        # 视角调整
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()