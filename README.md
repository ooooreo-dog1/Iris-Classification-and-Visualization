# -🌸 鸢尾花数据分类与可视化 (Iris Data Classification and Visualization)
📖 项目简介 (Project Overview)本项目是基于 Python 和 Scikit-learn 库实现的机器学习分类可视化实验，旨在通过图形化的方式深入理解不同分类器的决策机制、特征选择的重要性以及高维数据可视化方法。项目使用了经典的 Iris 鸢尾花数据集，重点在于生成各种决策边界、类别概率图以及 3D 决策超平面。

📁 文件结构 (File Structure)文件名描述对应实验任务data_preview.py数据探索。生成鸢尾花各特征的箱线图，用于初步分析特征分布和类别分离度。
特征探索all features classify.py任务四。可视化所有 $C_4^2=6$ 种特征组合下的 Logistic Regression 决策边界，用于特征选择。2D 决策边界对比classifier2d.py任务一。在最佳特征对上（花瓣特征），详细对比 多种分类器 的决策边界和类别概率热图。
2D 多分类器对比3d boundary.py任务二。使用 Logistic Regression 绘制三维特征空间中的线性决策超平面。
3D 线性可视化3D Probability Map.py任务三。使用 Gaussian Process Classifier 和切片法绘制三维特征空间中的非线性概率曲面。3D 非线性可视化project3.pptx实验项目演示文稿 (PPT)。演示材料深圳大学实验报告模板 (1).docx实验报告模板。报告模板README.md本文件。
-🛠️ 环境要求 (Prerequisites)本项目需要安装 Python 及其以下主要库：Python 3.6+scikit-learn (机器学习库)numpy (科学计算)matplotlib (基础绘图，用于 all features classify.py 等)seaborn (统计绘图，用于 data_preview.py)plotly (交互式绘图，用于 3D Probability Map.py)pandas (数据处理)您可以使用 pip 安装所有依赖项：Bashpip install scikit-learn numpy matplotlib seaborn pandas plotly
🚀 运行指南 (How to Run)您可以单独运行每个 Python 文件，以生成对应的可视化结果。1. 数据探索 (data_preview.py)运行此文件将显示各特征的箱线图，帮助确定最具区分力的特征。Bashpython data_preview.py
2. 2D 决策边界对比 (all features classify.py)运行此文件将生成包含 6 个子图的图像，展示不同特征组合下的 Logistic Regression 决策边界。Bashpython all features classify.py
3. 多分类器概率图 (classifier2d.py)运行此文件将生成一个复杂的 2D 图，对比五种分类器（包括逻辑回归、决策树、K近邻、SVM）的概率热图和最终决策边界。Bashpython classifier2d.py
4. 3D 可视化任务运行以下文件将分别生成 3D 线性超平面图和 3D 非线性概率曲面图。Bash# 3D 线性超平面
python "3d boundary.py"

# 3D 非线性概率曲面 (需 Plotly 支持)
python "3D Probability Map.py"
📜 实验报告 (Experiment Report)实验结果和分析已汇总在配套的实验报告中。请参考报告文件中的 Implementations 和 Results 部分以获得详细的理论和可视化分析。
