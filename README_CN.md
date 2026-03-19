# VSNet 快速部署指南

## 📖 项目简介
VSNet是一个用于亚毫米级相机位姿估计的孪生卷积神经网络实现。

原始论文: [Siamese Convolutional Neural Network for Sub-millimeter-accurate Camera Pose Estimation and Visual Servoing](https://arxiv.org/pdf/1903.04713.pdf)

## 🔧 环境准备

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载数据集
从以下链接下载训练数据集：
https://drive.google.com/file/d/1e3rykUUJltm5y-H3AF_lm-JMXOoiVgsh/view?usp=sharing

解压后放入 `./data` 目录

### 3. 修改配置
编辑 `config.py` 文件，修改数据路径：
```python
root_dir = './data'  # 改为你的数据目录
set_list = ['rotate']  # 你的数据集名称
```

## 🚀 快速开始

### 检查环境
```bash
python quick_start.py
```

### 训练模型
```bash
python train.py
```

### 推理测试
```python
from inference import VSNet

# 加载模型
model_path = './model.pth'
vsnet = VSNet(model_path, use_gpu=True)

# 推理两张图像
img_a = './test_img/image_a.png'
img_b = './test_img/image_b.png'
result = vsnet.infer(img_a, img_b)

# result格式: [tx, ty, tz, qw, qx, qy, qz]
# 前3个是平移向量，后4个是四元数
print('预测位姿:', result)
```

## 📁 项目结构

```
VSNet/
├── model.py              # 网络模型定义
├── train.py              # 训练脚本
├── inference.py          # 推理接口
├── dataset.py            # 数据加载器
├── utils.py              # 工具函数
├── transformations.py    # 3D变换库
├── data_aug.py          # 数据增强
├── config.py            # 配置文件 (新增)
├── quick_start.py       # 快速启动 (新增)
├── requirements.txt     # 依赖包 (新增)
└── README_CN.md         # 中文说明 (本文件)
```

## ⚙️ 主要配置项

在 `config.py` 中可以修改：
- `root_dir`: 数据集根目录
- `batch_size`: 批次大小 (默认256)
- `num_epochs`: 训练轮数 (默认10)
- `learning_rate`: 学习率 (默认0.0001)
- `img_size`: 图像尺寸 (默认640x480)

## 📊 训练监控

训练过程使用TensorBoard记录：
```bash
tensorboard --logdir=./results
```

## 🎯 模型输出

模型输出7维向量表示相对位姿：
- `[0:3]`: 平移向量 (x, y, z) 单位:米
- `[3:7]`: 旋转四元数 (w, x, y, z)

## 🔍 常见问题

### 1. CUDA out of memory
减小 `batch_size` 在 config.py 中

### 2. 找不到数据集
检查 `root_dir` 和 `set_list` 配置是否正确

### 3. 导入错误
运行 `pip install -r requirements.txt` 安装所有依赖

## 📝 引用

如果使用本代码，请引用原始论文：
```bibtex
@inproceedings{YuCai2019VS,
    title={Siamese Convolutional Neural Network for Sub-millimeter-accurate Camera Pose Estimation and Visual Servoing},
    author={Cunjun Yu and Zhongang Cai and Hung Pham and Quang-Cuong Pham},
    booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2019}
}
```
