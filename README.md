# 机器学习实验仓库

## 仓库简介
本仓库包含机器学习相关的实验代码、实验报告和结果文件。

## 目录结构
```
.
├── code/                      # 代码目录
│   ├── app.py                # Flask应用
│   ├── cnn_train.py          # CNN训练脚本
│   ├── gradio_app.py         # Gradio界面应用
│   ├── plot_loss_comparison.py  # 损失对比图
│   ├── plot_loss_curve.py    # 损失曲线图
│   ├── train.py              # 训练脚本
│   ├── train.csv             # 训练数据
│   ├── test.csv              # 测试数据
│   └── sample_submission.csv # 提交样例
│
├── report/                   # 实验报告目录
│   ├── CNN手写数字识别实验报告.md   # CNN实验报告
│   ├── CNN手写数字识别实验模板.md  # 实验模板
│   └── 手写数字识别系统搭建.pdf    # 系统搭建文档
│
├── results/                  # 实验结果目录
│   ├── loss_curve.png        # 损失曲线图
│   └── sample_submission.csv # 提交结果样例
│
└── README.md                 # 本文件

```

## 实验内容

### 1. CNN手写数字识别
- 使用PyTorch构建CNN模型
- 使用Flask/Gradio构建Web应用
- 训练和评估模型性能

### 2. 情感分析
- 使用词袋模型和TF-IDF进行文本特征提取
- 使用随机森林进行情感分类
- 测试句子："这款机器发热很快"

### 3. Word2Vec词向量与聚类
- 使用Word2Vec训练词向量模型
- 对词汇进行K-means聚类分析
- 聚成5个类别，分析每个类别的特点

## 环境配置
```bash
pip install torch torchvision
pip install flask gradio
pip install scikit-learn pandas numpy
pip install jieba gensim wordcloud
```

## 使用方法

### 训练模型
```bash
python code/train.py
```

### 启动Web应用
```bash
python code/app.py  # Flask应用
python code/gradio_app.py  # Gradio应用
```

## 作者
学号：112304260150
姓名：郭珉

## 许可
MIT License
