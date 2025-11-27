# Oil-Type Discrimination Using Enzyme Absorbance Signals

本项目利用 **类过氧化酶材料（15 种）** 与 **油类（13 种）** 的特异性反应所产生的 **吸光度特征** 来实现油类分类鉴别。  
机器学习模型使用多种经典分类器（DecisionTree / KNN / LDA / MLP / CatBoost / ExtraTrees / GaussianNB）以及 Voting Ensemble。

项目代码参考结构：cs230-code-examples，并做了工程化拆分。

## 🔬 1. 实验背景

- 材料（酶）：15 种
- 油类：13 种
- 实验设计：  
  **15 材料 × 13 油类 × 6 平行组 = 1170 个吸光度样本**  
  （可扩展到 9 组平行，共 1755 条）

记录的变量：
| 变量 | 描述 |
|------|------|
| en1–en15 | 15 种酶材料对应的吸光度值 |
| cate | 油类标签（13 种） |

最终目标：给盲样输入（吸光值 + 材料），模型自动预测油类类别。

## 📁 2. 项目结构
```plaintext
oil-type-discrimination/
├── README.md
├── requirements.txt
├── data/
│ ├── raw/
│ │ └── data-923.csv
│ └── processed/
├── notebooks/
│ └── iol_classification.ipynb
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── models.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
├── results/
│ ├── metrics/
│ ├── figures/
│ └── predictions/
├── models/
└── .gitignore
```

## 🧪 3. 安装依赖

Python 版本建议：**3.8–3.10**

```bash
pip install -r requirements.txt
```

## 📊 4. 数据准备

将原始实验数据放入：data/raw/data-923.csv
系统会自动：
- LabelEncoder 转换油类标签
- StandardScaler 进行特征标准化

## 🏋️ 5. 模型训练

python src/train.py

输出示例（不同模型准确率）：
模型：GaussianNB
- accuracy: 1.0
- precision: 1.00
- recall: 1.00
- F1: 1.00

模型：VotingClassifier
accuracy: 1.00


## 📈 6. 性能评估

查看某个模型的详细分类报告：
```
python src/evaluate.py --model models/best_model.pkl
```
会输出：
- Precision / Recall / F1
- 混淆矩阵（可选）
- 类别性能

## 🔍 7. 盲样预测（核心用途）
将盲样放入 CSV：blind_sample.csv
并预测油类：
```
python src/predict.py --model models/best_model.pkl --input blind_sample.csv
```
输出：results/predictions/pred_result.csv

## 🧪 8. Notebook（EDA）
项目包含：notebooks/iol_classification.ipynb,对应文章的算法和可视化代码

内容包括：
- 模型训练
- 可视化
  - Feature Correlation Matrix → 展示高相关特征，说明 LDA 合理性
  - LDA Scree Plot → 前 3 维覆盖 80% 变异，说明降维选择!
  - GaussianNB 2D/3D Prediction → 前 2 维坐标 + 模型预测标注，突出预测效果
  - ROC 曲线 → 模型判别能力量化
 

## 📘 9. 依赖
- scikit-learn
- xgboost
- lightgbm
- catboost
- pandas / numpy
- matplotlib / seaborn（仅 EDA）
