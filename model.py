import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torchvision.models as models  # 导入PyTorch的预训练模型模块


def loss_quat(output, target):  # 四元数损失函数，只考虑平移部分
    # compute rmse for translation - 计算平移部分的均方根误差
    trans_rmse = torch.sqrt(torch.mean((output[:, :3] - target[:, :3]) ** 2, dim=1))  # 计算平移误差的均方根

    return torch.mean(trans_rmse)  # 返回平移误差的平均值


def combined_loss_quat(output, target, weights=[1 / 2, 1 / 2]):  # 组合损失函数，同时考虑平移和旋转
    # compute rmse for translation - 计算平移部分的均方根误差
    trans_rmse = torch.sqrt(torch.mean((output[:, :3] - target[:, :3]) ** 2, dim=1))  # 计算平移误差的均方根

    # normalize quaternions - 归一化四元数
    normalized_quat = output[:, 3:] / torch.sqrt(torch.sum(output[:, 3:] ** 2, dim=1, keepdim=True))  # 将预测四元数归一化

    # compute rmse for rotation - 计算旋转部分的均方根误差
    quat_rmse = torch.sqrt(torch.mean((normalized_quat - target[:, 3:]) ** 2, dim=1))  # 计算旋转误差的均方根

    return torch.mean(weights[0] * trans_rmse + weights[1] * quat_rmse)  # 返回加权组合损失


def tf_dist_loss(output, target, device, unit_length=0.1):  # 变换距离损失函数
    """
    计算变换矩阵之间的几何距离损失
    
    参数:
        output: 模型预测的变换矩阵，形状为(batch_size, 12)或(batch_size, 3, 4)
               表示从图像b到图像a的预测相对位姿变换
        target: 真实的变换矩阵，形状与output相同
                表示从图像b到图像a的真实相对位姿变换
        device: 计算设备(torch.device)，如'cuda'或'cpu'
        unit_length: 单位长度值(默认0.1米)，用于创建测试向量
                   表示在x,y,z轴上各偏移unit_length的距离
    
    返回:
        平均损失值，表示预测变换与真实变换之间的几何差异
    """
    loss = torch.zeros([output.shape[0], 1])  # 初始化损失张量，形状为(batch_size, 1)
    unit_vec = torch.tensor([unit_length, unit_length, unit_length, 0.0]).view(4, 1).to(device)  # 创建测试向量[ux, uy, uz, 1]
    add_row = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 4).to(device)  # 创建添加行[0, 0, 0, 1]，用于将3x4矩阵扩展为4x4

    for i in range(output.shape[0]):  # 遍历批次中的每个样本
        output_slice = output[i].view(3, 4)  # 重塑输出为3x4矩阵
        output_mat = torch.cat((output_slice, add_row), dim=0)  # 构造4x4齐次变换矩阵

        # 计算真实变换与预测变换之间的相对变换，然后应用到测试向量上
        transformed = torch.matmul(torch.matmul(target[i, :].inverse(), output_mat), unit_vec)  # 计算变换后的单位向量

        # 计算变换后向量与原始向量之间的欧氏距离作为损失
        loss[i] = torch.sqrt(torch.sum((transformed[:3] - unit_vec[:3]) ** 2))  # 计算变换前后的欧氏距离

    return torch.mean(loss)  # 返回批次中所有样本的平均损失

class VSNet(nn.Module):  # VSNet视觉里程计网络类

    def __init__(self, num_classes=2):  # 初始化函数，参数为输出类别数
        super(VSNet, self).__init__()  # 调用父类初始化
        self.caffenet = models.alexnet(pretrained=True)  # 加载预训练的AlexNet模型

        self.caffenet.classifier[1] = nn.Linear(14 * 19 * 96 * 2, 4096)  # 修改分类器的第一层全连接层
        self.caffenet.classifier[-1] = nn.Linear(4096, 1024)  # 修改分类器的最后一层全连接层
        self.channelRed = nn.Conv2d(256, 96, 1)  # 通道减少层，将256通道减少到96通道
        self.output = nn.Sequential(  # 输出层序列
            nn.Linear(1024, 1024),  # 全连接层
            nn.Linear(1024, 1024),  # 全连接层
            nn.Linear(1024, 1024),  # 全连接层
            nn.Linear(1024, 1024),  # 全连接层
            nn.Linear(1024, num_classes))  # 最后一层全连接层，输出类别数

    def weight_init(self):  # 权重初始化函数
        for mod2 in self.output:  # 遍历输出层的所有模块
            if isinstance(mod2, nn.Conv2d) or isinstance(mod2, nn.Linear):  # 如果是卷积层或全连接层
                torch.nn.init.xavier_uniform_(mod2.weight)  # 使用Xavier均匀分布初始化权重
        torch.nn.init.xavier_uniform_(self.channelRed.weight)  # 初始化通道减少层的权重
        torch.nn.init.xavier_uniform_(self.caffenet.classifier[1].weight)  # 初始化分类器第一层的权重
        torch.nn.init.xavier_uniform_(self.caffenet.classifier[-1].weight)  # 初始化分类器最后一层的权重

    def forward(self, a, b):  # 前向传播函数，输入为两张图像
        # 256--96--14*19*96
        a = self.caffenet.features(a)  # 第一张图像通过特征提取器
        a = self.channelRed(a)  # 减少通道数
        a = a.view(a.size(0), 14 * 19 * 96)  # 展平特征

        b = self.caffenet.features(b)  # 第二张图像通过特征提取器
        b = self.channelRed(b)  # 减少通道数
        b = b.view(b.size(0), 14 * 19 * 96)  # 展平特征

        # 14 * 19 * 96 * 2 = 51,072
        concat = torch.cat((a, b), 1)  # 连接两张图像的特征
        # 14 * 19 * 96 * 2 = 51,072---4096--4096--1024
        match = self.caffenet.classifier(concat)  # 通过分类器
        match = self.output(match)  # 通过输出层

        return match  # 返回匹配结果