import time  # 导入时间模块
import datetime  # 导入日期时间模块
import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import cv2  # 导入OpenCV库，用于图像处理
import os  # 导入操作系统模块，用于文件和目录操作
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from torch.optim.lr_scheduler import MultiStepLR  # 导入多步学习率调度器
from torch.utils.data import DataLoader  # 导入数据加载器

import tensorboard_logger as tb  # 导入TensorBoard日志记录器

import dataset  # 导入自定义数据集模块
from model import VSNet, loss_xy  # 从模型模块导入VSNet类和组合损失函数
from utils import check_dir, axis_angle_from_quat, normalize_q, get_stem, accuracy_thres_curve  # 导入工具函数
from transformations import angle_between_vectors, euler_from_quaternion  # 导入变换相关函数

model_name = 'VSNet-AF-400train-7'  # 模型名称
batch_size = 1  # 批处理大小
GPU_IDS = [0]  # GPU设备ID列表，可以设置多个GPU如[0, 1, 2, 3]
# 运行模式配置
mode = ('train', 'train')  # 运行模式：训练训练集
# mode = ('eval', 'train')  # 评估训练集
# mode = ('eval', 'dev')  # 评估验证集
# mode = ('eval', 'test')  # 评估测试集

# 模型配置参数
model_pretrained = None  # 预训练模型路径，None表示不使用预训练
num_classes = 2  # 输出类别数，X和Y

# 数据集配置参数
root_dir = './AngFeng_Dataset'  # 数据根目录
pattern = 'camera'  # 数据集模式，表示六自由度误差在1cm和5度范围内
set_list = ['1'] # 数据集列表，可以包含多个数据集如['1', '2', '3']

# 构建图像和标签目录列表
img_dir_list = [root_dir + '/' + pattern + my_set + '/img' for my_set in set_list]  # 图像目录列表
label_dir_list = [root_dir + '/' + pattern + my_set + '/label' for my_set in set_list]  # 标签目录列表

# 保存配置参数
save_root_dir = './results'  # 保存根目录
log_name = save_root_dir + '/' + model_name + '/' + 'log.txt'  # 日志文件路径
img_size = (1024, 1280)   #输入图像尺寸,高度,宽度

# 数据集大小配置
train_size_list = [360] * len(set_list)  # 每个数据集的训练样本数量
dev_size_list = [20] * len(set_list)  # 每个数据集的验证样本数量
test_size_list = [20] * len(set_list)  # 每个数据集的测试样本数量

# 训练配置参数
num_epochs = 10  # 训练轮数
aug_factor = 1  # 数据增强因子，增强比例为25%
num_workers = 8  # 数据加载的工作进程数

# 优化器配置参数
learning_rate = 1e-4  # 学习率
milestones = [int(num_epochs * 0.4), int(num_epochs * 0.6), int(num_epochs * 0.8)]  # 学习率调整的关键点
# milestones = list(range(num_epochs))  # 可选：每个epoch都调整学习率
gamma = 0.5  # 学习率衰减倍数
momentum = 0.9  # 动量参数
limits = None  # 偏差限制，None表示不限制
weights = [1, 0]  # 损失权重，平移和旋转的权重分配
weight_decay = 1e-4  # 权重衰减，L2正则化参数

random_seed = 2  # 随机种子，确保结果可复现

# 断点保存和恢复配置
resume_training = False  # 是否从断点恢复训练
checkpoint_interval = 1  # 每隔多少个epoch保存一次断点
checkpoint_path = save_root_dir + '/' + model_name + '/checkpoint.pth'  # 断点文件路径

# 早停配置
early_stopping_enabled = True  # 是否启用早停
early_stopping_patience = 10  # 验证集指标连续多少个epoch不提升后停止
early_stopping_min_delta = 1e-4  # 判定为提升的最小变化量
best_model_path = save_root_dir + '/' + model_name + '/best_model.pth'  # 最优模型路径

def prepare_loaders(return_path=False):  # 准备数据加载器的函数
    """
    准备训练、验证和测试数据加载器
    
    参数:
        return_path: 是否返回图像路径，用于评估时可视化错误样本
    
    返回:
        train_loader: 训练数据加载器
        dev_loader: 验证数据加载器
        test_loader: 测试数据加载器
        train_size_aug: 增强后的训练集大小
        dev_size_aug: 增强后的验证集大小
        test_size_aug: 增强后的测试集大小
    """
    # 使用dataset模块的split_sets函数分割数据集
    train_set_paths, dev_set_paths, test_set_paths = dataset.split_sets(img_dir_list=img_dir_list,
                                                                            label_dir_list=label_dir_list,
                                                                            train_size_list=train_size_list,
                                                                            dev_size_list=dev_size_list,
                                                                            test_size_list=test_size_list,
                                                                            random_seed=random_seed,
                                                                            aug_factor=aug_factor,
                                                                            limits=limits, weights=weights)

    # 创建数据集实例
    train_set = dataset.VSDataset(set_paths=train_set_paths, img_size=img_size, return_path=return_path)
    dev_set = dataset.VSDataset(set_paths=dev_set_paths, img_size=img_size, return_path=return_path)
    test_set = dataset.VSDataset(set_paths=test_set_paths, img_size=img_size, return_path=return_path)

    # 获取增强后的数据集大小
    train_size_aug = len(train_set)
    dev_size_aug = len(dev_set)
    test_size_aug = len(test_set)

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, dev_loader, test_loader, train_size_aug, dev_size_aug, test_size_aug


def save_checkpoint(model, optimizer, scheduler, epoch, tb_count, checkpoint_path):
    """
    保存训练断点
    
    参数:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        tb_count: TensorBoard计数器
        checkpoint_path: 断点保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'tb_count': tb_count
    }
    torch.save(checkpoint, checkpoint_path)
    print('Checkpoint saved at {}'.format(checkpoint_path))


def load_checkpoint(checkpoint_path, device):
    """
    加载训练断点
    
    参数:
        checkpoint_path: 断点文件路径
        device: 设备
    
    返回:
        checkpoint: 断点数据
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print('Checkpoint loaded from {}'.format(checkpoint_path))
    print('Resuming from epoch {}'.format(checkpoint['epoch']))
    return checkpoint


def mode_train(train_loader, dev_loader, train_size_aug, dev_size_aug):  # 训练模式函数
    """
    训练模型并在验证集上评估
    
    参数:
        train_loader: 训练数据加载器
        dev_loader: 验证数据加载器
        train_size_aug: 增强后的训练集大小
        dev_size_aug: 增强后的验证集大小
    """
    check_dir(save_root_dir + '/' + model_name)  # 检查并创建保存目录

    device = torch.device('cuda:{}'.format(GPU_IDS[0]))  # 设置计算设备为指定的GPU

    # 加载或创建模型
    if model_pretrained:  # 如果有预训练模型
        print('Loading pretrained model from {}'.format(save_root_dir + '/' + model_pretrained + '/model.pth'))
        model = torch.load(save_root_dir + '/' + model_pretrained + '/model.pth', map_location=device)
    else:  # 如果没有预训练模型，创建新模型
        model = VSNet(num_classes=num_classes)
    if len(GPU_IDS) > 1:
        model = nn.DataParallel(model, device_ids=GPU_IDS)  # 多GPU使用DataParallel

    # criterion = nn.MSELoss(reduction='sum')  # 可选的损失函数
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # 可选的SGD优化器

    model.to(device)  # 将模型移动到GPU

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)  # 创建学习率调度器

    tb.configure(save_root_dir + '/' + model_name)  # 配置TensorBoard日志

    start_time = time.time()  # 记录开始时间

    tb_count = 0  # TensorBoard计数器
    start_epoch = 0  # 起始epoch
    best_dev_score = np.inf
    no_improve_count = 0
    
    # 检查是否从断点恢复训练
    if resume_training:
        if os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(checkpoint_path, device)
            
            # 恢复模型状态
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # 恢复优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复学习率调度器状态
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复训练状态
            start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
            tb_count = checkpoint['tb_count']
            
            print('Resumed training from epoch {}'.format(start_epoch))
        else:
            print('Checkpoint file not found at {}. Starting from scratch.'.format(checkpoint_path))
    
    for epoch in range(start_epoch, num_epochs):  # 从起始epoch开始遍历

        # Training - 训练阶段
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 初始化累积损失
        for i, sample in enumerate(train_loader, 0):  # 遍历训练数据
            if i == 1 and epoch == 0:  # 从第二个batch开始计时，避免初始化影响
                start_time = time.time()
            img_a, img_b, label = sample  # 获取图像对和标签

            optimizer.zero_grad()  # 清零梯度

            # 将数据移动到GPU
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            label = label.to(device)

            output = model(img_a, img_b)  # 前向传播

            loss = loss_xy(output, label)  # 计算损失

            loss.backward()  # 反向传播

            optimizer.step()  # 更新参数

            running_loss += loss.item() * output.shape[0]  # 累积损失

            # 将数据移回CPU用于计算误差
            output = output.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            error = np.zeros(2)  # 初始化误差数组

            # 计算每个样本的误差
            for j in range(output.shape[0]):
                # 平移误差(前2个元素)
                error[:2] += np.abs(output[j, :2] - label[j, :2])

            # 计算平均误差
            error /= output.shape[0]
            # 估算剩余时间
            est_time = (time.time() - start_time) / (epoch * len(train_loader) + i + 1) * (
                    num_epochs * len(train_loader))
            est_time = str(datetime.timedelta(seconds=est_time))
            # 打印训练信息
            print(
                '[TRAIN][{}][EST:{}] Epoch {}, Batch {}, Loss = {:0.7f}, error: x={:0.2f}mm,y={:0.2f}mm'.format(
                    time.time() - start_time, est_time, epoch + 1, i + 1,
                    loss.item(), *error))

            # 记录到TensorBoard
            tb.log_value(name='Loss', value=loss.item(), step=tb_count)
            tb.log_value(name='x/mm', value=error[0], step=tb_count)
            tb.log_value(name='y/mm', value=error[1], step=tb_count)
            tb_count += 1
            
        # Dev eval - 验证集评估
        model.eval()  # 设置模型为评估模式，禁用dropout等训练时特有的层
        with torch.no_grad():  # 禁用梯度计算，减少内存消耗并加速计算
            running_error_dev = np.zeros(2)  # 初始化验证集累积误差数组
            running_loss_dev = 0.0  # 初始化验证集累积损失
            # running_error_dev = np.zeros(2)  # 可选：只计算部分误差
            for i, sample in enumerate(dev_loader, 0):  # 遍历验证数据
                img_a, img_b, label = sample  # 获取图像对和标签

                # 将数据移动到GPU
                img_a = img_a.to(device)
                img_b = img_b.to(device)
                label = label.to(device)

                output = model(img_a, img_b)  # 前向传播，获取预测结果
                loss = loss_xy(output, label)  # 计算验证集损失
                running_loss_dev += loss.item() * output.shape[0]  # 累积验证集损失

                # 将数据移回CPU用于计算误差
                output = output.cpu().detach().numpy()
                label = label.cpu().detach().numpy()

                error = np.zeros(2)  # 初始化当前批次的误差数组

                # 计算每个样本的误差
                for j in range(output.shape[0]):
                    # 计算平移误差(前2个元素)
                    error[:2] += np.abs(output[j, :2] - label[j, :2])

                running_error_dev += error  # 累积误差
                error /= output.shape[0]  # 计算当前批次的平均误差

                # 打印验证信息
                print(
                    '[EVAL][{}] Epoch {}, Batch {}, Loss={:0.7f}, error: x={:0.2f}mm,y={:0.2f}mm'.format(
                        time.time() - start_time, epoch + 1, i + 1, loss.item(), *error))

        # 计算平均损失和误差
        average_loss = running_loss / train_size_aug  # 计算平均训练损失
        average_dev_loss = running_loss_dev / dev_size_aug  # 计算平均验证损失
        average_error = running_error_dev / dev_size_aug  # 计算平均验证误差
        # 打印总结信息
        print(
            '[SUMMARY][{}] Summary: Epoch {}, Train Loss={:0.7f}, Dev Loss={:0.7f}, Dev Error: x={:0.2f}mm,y={:0.2f}mm\n'.format(
                time.time() - start_time, epoch + 1, average_loss, average_dev_loss, *average_error))

        # 记录到TensorBoard
        tb.log_value(name='Train loss', value=average_loss, step=epoch)  # 记录训练损失
        tb.log_value(name='Dev loss', value=average_dev_loss, step=epoch)  # 记录验证损失
        tb.log_value(name='Dev x/mm', value=average_error[0], step=epoch)  # 记录x轴平移误差
        tb.log_value(name='Dev y/mm', value=average_error[1], step=epoch)  # 记录y轴平移误差

        scheduler.step()  # 更新学习率

        # 保存loss和error数据为txt文件
        results_dir = save_root_dir + '/' + model_name
        with open(results_dir + '/Val_log.txt', 'a') as f:
            f.write('Epoch {}, Loss = {:0.7f}, Dev Error: x={:0.2f}mm, y={:0.2f}mm\n'.format(
                epoch + 1, average_dev_loss, average_error[0], average_error[1]))
        print('Val log saved at {}/Val_log.txt'.format(results_dir))
 
        # 保存断点
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, tb_count, checkpoint_path)

        # 早停逻辑：以dev x/y误差之和作为监控指标（越小越好）
        if early_stopping_enabled:
            # 计算当前epoch的验证集评分（x轴误差 + y轴误差）
            current_dev_score = float(average_error[0] + average_error[1])
            
            # 检查当前评分是否比最佳评分有明显改善
            # 只有改善量大于early_stopping_min_delta才算真正改善
            if current_dev_score < best_dev_score - early_stopping_min_delta:
                # 更新最佳验证集评分
                best_dev_score = current_dev_score
                # 重置未改善计数器
                no_improve_count = 0
                # 保存当前模型为最佳模型
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), best_model_path)
                # 打印模型更新信息
                print('Best model updated at {}, dev score={:0.6f}'.format(best_model_path, best_dev_score))
                with open(results_dir + '/Val_log.txt', 'a') as f:
                    f.write('Best model updated at Epoch {}, dev score={:0.6f}\n'.format(epoch + 1, best_dev_score))
            else:
                # 验证集评分没有改善，增加未改善计数器
                no_improve_count += 1
                # 打印早停计数器状态
                print('Early stopping counter: {}/{}'.format(no_improve_count, early_stopping_patience))
                
                # 检查是否达到早停条件
                # 如果连续early_stopping_patience个epoch都没有改善，则触发早停
                if no_improve_count >= early_stopping_patience:
                    # 打印早停触发信息
                    print('Early stopping triggered at epoch {}.'.format(epoch + 1))
                    # 跳出训练循环，停止训练
                    break

def mode_eval(loader, size_aug):  # 评估模式函数
    """
    在指定数据集上评估模型性能
    
    参数:
        loader: 数据加载器
        size_aug: 数据集大小
    """

    # 精度-阈值曲线配置
    make_curve = True  # 是否生成精度-阈值曲线
    if make_curve:
        xy_error_max = 10.0  # mm - 平移误差最大值
        xy_error_reso = 0.01  # mm - 平移误差分辨率

    device = torch.device('cuda:{}'.format(GPU_IDS[0]))  # 设置计算设备为指定的GPU

    # 创建模型实例
    model = VSNet(num_classes=num_classes)
    
    # 加载模型状态字典
    if os.path.exists(best_model_path):
        model_state_dict = torch.load(best_model_path, map_location=device)
    else:
        model_state_dict = torch.load(checkpoint_path, map_location=device)['model_state_dict']
    model.load_state_dict(model_state_dict)

    # 如果需要多GPU评估
    if len(GPU_IDS) > 1:
        model = nn.DataParallel(model, device_ids=GPU_IDS)

    model.eval()  # 设置模型为评估模式
    model.to(device)  # 将模型移动到GPU

    data = [[] for i in range(8)]  # 用于箱线图的数据
    running_error_test = np.zeros(8)  # 初始化测试集累积误差
    start_time = time.time()  # 记录开始时间
    for i, sample in enumerate(loader):  # 遍历测试数据

        # 获取数据和路径
        img_a, img_b, label, img_a_path, img_b_path, label_a_path, label_b_path = sample
        # 将数据移动到GPU
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        label = label.to(device)

        output = model(img_a, img_b)  # 前向传播
        # 将数据移回CPU用于计算误差
        output = output.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        # print('output = \n{}\nlabel = \n{}'.format(output, label))

        error = np.zeros(8)  # 初始化误差数组
        for j in range(output.shape[0]):  # 遍历批次中的每个样本

            # print('{} vs {}'.format(output[j], label[j]))

            # 计算平移误差(转换为毫米)
            xy_error = np.abs(output[j, :2] - label[j, :2])
            error[:2] += xy_error
         
            # 收集数据用于箱线图
            data[0].append(xy_error[0])  # x轴平移误差
            data[1].append(xy_error[1])  # y轴平移误差

        running_error_test += error  # 累积误差
        error /= output.shape[0]  # 计算平均误差

        # 打印评估信息
        print(
            '[EVAL][{}] Batch {}, error: x={:0.2f}mm,y={:0.2f}mm'.format(
                time.time() - start_time, i + 1, *error))

    # 计算平均误差
    average_error = running_error_test / test_size_aug
    print(
        'Summary: test_eval: x={:0.2f}mm,y={:0.2f}mm\n\n'.format(
        *average_error))
        
    # 保存测试结果到txt文件
    results_dir = save_root_dir + '/' + model_name
    with open(results_dir + '/test_results.txt', 'w') as f:
        f.write('Test Results Summary\n')
        f.write('===================\n')
        f.write('Test size: {}\n'.format(test_size_aug))
        f.write('Average Error: x={:0.2f}mm, y={:0.2f}mm\n'.format(average_error[0], average_error[1]))
    print('Test results saved at {}/test_results.txt'.format(results_dir))
    
        
    # 创建误差分布箱线图
    fig1 = plt.figure(0)
    ax11 = fig1.add_subplot(121)  # 左子图：平移误差
    ax11.set_title('Translation Errors (mm)')
    bp11 = ax11.boxplot(data[:2])  # 绘制x,y平移误差的箱线图
    ax11.set_xticklabels(['x', 'y'])
    # 只显示最大异常值
    for outliers in bp11['fliers']:
        outliers.set_data([[outliers.get_xdata()[0]], [[np.max(outliers.get_ydata())]]])

    plt.savefig(save_root_dir + '/' + model_name + '/error_distribution.png')  # 保存误差分布图
    plt.show()  # 显示图像

    # 生成精度-阈值曲线
    if make_curve:
        # 创建阈值列表
        xy_thres_list = list(np.arange(0, xy_error_max, xy_error_reso))

        # 计算每个误差维度的精度-阈值曲线
        x_accuracy_list, x_thres_list = accuracy_thres_curve(data[0], xy_thres_list)
        y_accuracy_list, y_thres_list = accuracy_thres_curve(data[1], xy_thres_list)
      
        # 创建精度-阈值曲线图
        fig2 = plt.figure()
        ax21 = fig2.add_subplot(211)  # 上子图：平移误差
        ax21.set_xlabel('Threshold (mm)')
        ax21.set_ylabel('Fraction of pass')
        lines21 = ax21.plot(x_thres_list, x_accuracy_list, 'r-', y_thres_list, y_accuracy_list, 'g-')
        ax21.set_xticks(np.arange(min(x_thres_list), max(x_thres_list) + 1, 1.0))
        ax21.legend(lines21, ('x', 'y'))

        plt.tight_layout()  # 调整布局
        plt.savefig(save_root_dir + '/' + model_name + '/error_curve.png')  # 保存精度-阈值曲线
        plt.show()  # 显示图像

if __name__ == '__main__':  # 主程序入口

    if mode[0] == 'train':  # 如果是训练模式

        # 准备数据加载器（不需要返回路径）
        train_loader, dev_loader, test_loader, train_size_aug, dev_size_aug, test_size_aug = prepare_loaders(
            return_path=False)

        if mode[1] == 'train':  # 训练训练集
            mode_train(train_loader, dev_loader, train_size_aug, dev_size_aug)
        else:
            raise Exception('Cannot train ' + mode[1] + ' set.')  # 不支持训练其他数据集

    elif mode[0] == 'eval':  # 如果是评估模式

        # 准备数据加载器（需要返回路径用于可视化）
        train_loader, dev_loader, test_loader, train_size_aug, dev_size_aug, test_size_aug = prepare_loaders(
            return_path=True)

        if mode[1] == 'train':  # 评估训练集
            mode_eval(train_loader, train_size_aug)
        elif mode[1] == 'dev':  # 评估验证集
            mode_eval(dev_loader, dev_size_aug)
        elif mode[1] == 'test':  # 评估测试集
            mode_eval(test_loader, test_size_aug)
        else:
            raise Exception('Cannot eval ' + mode[1] + ' set.')  # 不支持评估其他数据集

    else:
        raise Exception('Unknown mode ' + mode[0])  # 不支持的模式
