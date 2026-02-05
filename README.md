# Oil-Type Discrimination Using Enzyme Absorbance Signals

> 基于**类过氧化酶材料**与**油类**特异性反应的吸光度特征，使用机器学习实现油类分类鉴别。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 项目概述

本项目利用 **15 种类过氧化酶材料** 与 **13 种油类** 的特异性反应所产生的 **吸光度特征**，构建了一个可复现的机器学习管道来实现油类分类鉴别。

**核心模型**: StandardScaler → LDA → Gaussian Naive Bayes

**性能指标**:
- 交叉验证准确率: **100%**
- Permutation Test p-value: **0.000999**

---

## 📊 实验设计

| 参数 | 数值 |
|------|------|
| 材料（酶） | 15 种 |
| 油类 | 13 种 (A-M) |
| 平行组 | 6 组 (可扩展至 9 组) |
| 样本总量 | 78 (13 × 6) |

**数据变量**:
- `en1–en15`: 15 种酶材料对应的吸光度值
- `cate`: 油类标签 (A, B, C, ..., M)

---

## 📁 项目结构

```
oil-type-discrimination/
│
├── data/                          # 数据目录
│   └── raw/
│       └── data-923.csv          # 原始数据 (78 samples × 15 features)
│
├── src/                           # 核心代码
│   ├── ml_model.py               # MLModel 类定义
│   ├── utils.py                  # 统一工具函数 (数据加载、模型创建等)
│   ├── config.py                 # 配置常量
│   ├── train.py                  # 训练脚本
│   ├── evaluate.py               # 评估脚本
│   ├── predict.py                # 预测脚本
│   ├── visualization/            # 可视化模块
│   │   ├── __init__.py
│   │   ├── nature_plots.py       # Nature 风格图表
│   │   ├── statistical_plots.py  # 统计图表 (ROC, 混淆矩阵等)
│   │   └── interactive_plots.py  # 3D 交互式可视化
│   └── validation/               # 验证框架
│       ├── validation.py
│       ├── leave_replicate_out.py
│       ├── multi_seed_cv.py
│       ├── feature_ablation.py
│       ├── leave_one_oil_out.py
│       ├── label_permutation_test.py
│       ├── run_all_validations.py
│       └── README.md
│
├── notebooks/                     # Jupyter Notebooks
│   ├── iol_classification.ipynb  # 算法对比 + 论文图表
│   ├── model.ipynb               # 模型验证分析
│   └── output/                  # Notebook 生成的文件
│
├── scripts/                       # 实用脚本
│   └── generate_paper_figures.py # 一键生成所有论文图表
│
├── paper/                         # 论文相关文件
│   ├── figures/                  # 论文图表输出
│   └── README.md
│
├── experiments/                   # 实验结果
│   ├── base_model/
│   ├── leave_replicate_out/
│   ├── multi_seed/
│   ├── feature_ablation/
│   ├── leave_one_oil_out/
│   ├── label_permutation/
│   └── validation_report/
│
├── requirements.txt               # Python 依赖
└── README.md                      # 项目说明
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository_url>
cd oil-type-discrimination

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

将原始数据文件放置到 `data/raw/` 目录：

```bash
data/raw/data-923.csv
```

### 3. 运行示例

```bash
# 训练模型
python src/train.py

# 评估模型
python src/evaluate.py

# 运行完整验证框架 (证明模型泛化能力)
python src/validation/run_all_validations.py

# 查看验证报告
open experiments/validation_report/validation_report.md
```

---

## 📚 论文图表生成

### 方法 1: 使用统一脚本（推荐）

```bash
python3 scripts/generate_paper_figures.py
```

**输出目录**: `paper/figures/`

**生成文件**:

| 文件 | 类型 | 说明 |
|------|------|------|
| `fig1_nature_scatter_2d.pdf` | PDF | Nature 风格 2D 散点图 |
| `fig1_nature_scatter_2d.svg` | SVG | 矢量图 (可编辑) |
| `fig2_lda_scree.pdf` | PDF | LDA Scree 图 |
| `fig2_lda_scree_data.csv` | CSV | LDA Scree 数据 |
| `fig3_correlation_heatmap.pdf` | PDF | 特征相关性热图 |
| `fig4_roc_curves.pdf` | PDF | ROC 曲线对比 |
| `fig4_roc_auc_summary.csv` | CSV | AUC 汇总数据 |
| `fig4_roc_curves_data.csv` | CSV | ROC 曲线数据 |
| `fig5_3d_interactive.html` | HTML | 3D 交互式可视化 |
| `fig6_validation_summary.png` | PNG | 验证摘要 |

### 方法 2: 运行 Notebooks

```bash
# 安装 Jupyter
pip install jupyter plotly kaleido

# 启动 Jupyter
jupyter notebook

# 运行 notebooks (生成的文件保存到 notebooks/output/)
# - notebooks/iol_classification.ipynb
# - notebooks/model.ipynb
```

**Notebook 输出目录**: `notebooks/output/`

---

## 🔬 模型验证框架

项目包含完整的验证框架，用于证明模型的泛化能力：

### 运行所有验证

```bash
python3 src/validation/run_all_validations.py
```

### 单独运行各验证方法

| 验证方法 | 说明 | 脚本 |
|----------|------|------|
| **Leave-Replicate-Out CV** | 按平行组交叉验证 | `python src/validation/leave_replicate_out.py` |
| **Multi-Seed CV** | 多随机种子验证 + 置信区间 | `python src/validation/multi_seed_cv.py` |
| **Feature Ablation** | 特征消融研究 | `python src/validation/feature_ablation.py` |
| **Leave-One-Oil-Out** | 留一油法验证 | `python src/validation/leave_one_oil_out.py` |
| **Label Permutation** | 标签排列检验 | `python src/validation/label_permutation_test.py` |

### 验证报告

所有验证结果汇总在：
- `experiments/validation_report/validation_summary.png` - 可视化摘要
- `experiments/validation_report/validation_report.md` - 详细报告

---

## 📖 依赖项

**核心依赖**:
```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.0.0
scipy>=1.7.0
```

**可视化**:
```
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=5.0.0
kaleido>=0.2.0
```

**Notebook 支持**:
```
jupyter>=1.0.0
ipykernel>=6.0.0
```

**可选算法**:
```
catboost>=1.0.0
xgboost>=1.0.0
```

**完整安装**:
```bash
pip install -r requirements.txt
```

---

## 📝 Notebooks 说明

### `iol_classification.ipynb`
探索性数据分析、算法对比和论文图表生成。

**内容包括**:
- 算法对比 (Decision Tree, KNN, GaussianNB, MLP, LDA, CatBoost, Extra Trees)
- Nature 风格可视化
- 特征相关性分析
- LDA Scree Plot
- ROC 曲线对比
- 3D 交互式可视化

**生成文件**: 保存到 `notebooks/output/`

### `model.ipynb`
模型验证和泛化能力分析。

**内容包括**:
- 5-Fold 分层交叉验证
- Permutation Test (p-value = 0.000999)
- LDA Scree Plot

**生成文件**: 保存到 `notebooks/output/`

---

## 🛠️ 开发指南

### 代码规范

- **统一工具函数**: 所有脚本使用 `src/utils.py` 中的工具函数
- **导入路径**: 使用 `from src.xxx import yyy` 的相对导入方式
- **配置管理**: 使用 `src/config.py` 管理配置常量

### 添加新的可视化

在 `src/visualization/` 中添加新函数：

```python
# src/visualization/my_plots.py
import matplotlib.pyplot as plt

def plot_my_visualization(X, y, output_path):
    # Your plotting code here
    plt.savefig(output_path, dpi=300)
```

在 `src/visualization/__init__.py` 中导出：

```python
from .my_plots import plot_my_visualization
```

### 添加新的验证方法

在 `src/validation/` 中创建新脚本：

```python
# src/validation/my_validation.py
from src.utils import load_data, create_pipeline, save_results

def run_my_validation():
    X, y = load_data()
    # Your validation code here
    results = {...}
    save_results(results, "experiments/my_validation/")
```

---

## 📄 License

MIT License

---

## 🙏 致谢

本项目基于论文研究开发，感谢所有贡献者的支持。

---

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
