import torch  # 导入PyTorch深度学习框架
import torchvision  # 导入PyTorch的计算机视觉模块
import os  # 导入操作系统相关功能的模块
import numpy as np  # 导入NumPy库，用于数值计算
from torchvision import transforms  # 导入PyTorch的图像变换模块
from PIL import Image  # 从PIL库导入Image模块，用于图像处理
from torch.utils.data import Dataset  # 导入PyTorch数据集基类
from utils import get_files, get_stem  # 从自定义工具模块导入辅助函数
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器类
from model import VSNet, loss_xy  # 从模型模块导入VSNet类和组合损失函数
from dataset import load_xy_label  # 从数据集模块导入标签加载函数
import torch.nn as nn  # 导入PyTorch的神经网络模块
import time  # 导入时间模块，用于计时


model_name = 'VSNet-AF-400train-4'  # 模型名称

# 数据集配置参数
root_dir = './AngFeng_Dataset'  # 数据根目录
pattern = 'camera'  # 数据集模式，表示六自由度误差在1cm和5度范围内
set_list = ['1'] # 数据集列表，可以包含多个数据集如['1', '2', '3']
img_base = './AngFeng_Dataset/base.png'  # 基础图像路径
label_base_path = './AngFeng_Dataset/base.txt'  # 基础标签文件路径

# 构建图像和标签目录列表
img_dir_list = [root_dir + '/' + pattern + my_set + '/img' for my_set in set_list]  # 图像目录列表
label_dir_list = [root_dir + '/' + pattern + my_set + '/label' for my_set in set_list]  # 标签目录列表

# 保存配置参数
save_root_dir = './results'  # 保存根目录
log_name = save_root_dir + '/' + model_name + '/' + 'log.txt'  # 日志文件路径
img_size = (1280, 1024)  # 输入图像尺寸
best_model_path = save_root_dir + '/' + model_name + '/' + 'best_model.pth'  # 最佳模型路径
inference_results_dir = save_root_dir + '/' + model_name + '/' + 'inference_results'  # 推理结果保存目录
# 数据集大小配置
train_size_list = [360] * len(set_list)  # 每个数据集的训练样本数量
dev_size_list = [20] * len(set_list)  # 每个数据集的验证样本数量
test_size_list = [20] * len(set_list)  # 每个数据集的测试样本数量

random_seed = 2 # 随机种子，用于数据划分和模型训练的可重复性

batch_size = 1  # 批次大小
num_workers = 0  # 数据加载器的工作线程数，0表示在主

num_classes = 2  # 输出类别数，X和Y
GPU_IDS = [0]  # GPU设备ID列表，可以设置多个GPU如[0, 1, 2, 3]


def get_test_raw_samples(img_dir_list, label_dir_list,  # 函数定义，获取测试集原始样本（单帧图像和其标签路径）
                         train_size_list=[600], dev_size_list=[200], test_size_list=[200],  # 各数据集大小列表
                         random_seed=0,img_size=(640, 480),img_ext='.png', label_ext='.txt'):  # 随机种子和文件扩展名
    """
    获取测试集原始样本(单帧图像和其标签路径)，不做数据增强或配对。

    切分方式与split_sets保持一致：
    1) 每个目录先按文件名排序；
    2) 使用相同random_seed生成np.random.permutation；
    3) 先切train，再切dev，最后切test。
    """

    # 断言：确保所有输入都是列表类型
    assert isinstance(img_dir_list, list), 'img_dir_list must be a list!'  # 确保图像目录列表是列表
    assert isinstance(label_dir_list, list), 'label_dir_list must be a list!'  # 确保标签目录列表是列表
    assert isinstance(train_size_list, list), 'train_size_list must be a list!'  # 确保训练集大小列表是列表
    assert isinstance(dev_size_list, list), 'dev_size_list must be a list!'  # 确保验证集大小列表是列表
    assert isinstance(test_size_list, list), 'test_size_list must be a list!'  # 确保测试集大小列表是列表

    # 断言：确保所有列表的长度相等
    assert len(img_dir_list) == len(label_dir_list) == len(train_size_list) == len(dev_size_list) == len(test_size_list), \
        'lists must have equal lengths!'  # 确保所有列表长度相等

    # 初始化返回的测试样本列表（每个元素为(img_path, label_path)元组）
    ret_test_samples = []
    
    # 遍历所有数据目录
    for i in range(len(img_dir_list)):  # 遍历每个数据集目录
        img_dir = img_dir_list[i]  # 获取第i个图像目录
        label_dir = label_dir_list[i]  # 获取第i个标签目录
        train_size = train_size_list[i]  # 获取第i个数据集的训练集大小
        dev_size = dev_size_list[i]  # 获取第i个数据集的验证集大小
        test_size = test_size_list[i]  # 获取第i个数据集的测试集大小

        # 获取该目录下所有图像和标签文件的路径，并按文件名排序
        img_paths = sorted(get_files(img_dir, img_ext))  # 获取排序后的图像文件路径列表
        label_paths = sorted(get_files(label_dir, label_ext))  # 获取排序后的标签文件路径列表

        # 断言：确保图像和标签数量相等
        assert len(img_paths) == len(label_paths), 'Unequal number of images and labels found!'  # 图像和标签数量必须相等
        # 断言：确保有足够的样本用于分割
        assert len(img_paths) >= train_size + dev_size + test_size, 'Not enough images and labels are found'  # 确保有足够的样本

        # 设置随机种子，确保结果可复现
        np.random.seed(random_seed)  # 设置NumPy随机种子
        # 生成随机排列的索引
        perm_list = np.random.permutation(len(img_paths))  # 随机排列图像索引
        # 将路径列表转换为NumPy数组，便于索引操作
        img_paths = np.array(img_paths)  # 转换为NumPy数组
        label_paths = np.array(label_paths)  # 转换为NumPy数组

        # 计算测试集的索引范围
        test_start = train_size + dev_size  # 测试集起始索引
        test_end = test_start + test_size  # 测试集结束索引
        # 使用随机排列后的索引提取测试集路径
        test_img_paths = img_paths[perm_list[test_start:test_end]].tolist()  # 测试集图像路径列表
        test_label_paths = label_paths[perm_list[test_start:test_end]].tolist()  # 测试集标签路径列表
   
        # 将图像路径和标签路径打包成元组列表
        test_samples = list(zip(test_img_paths, test_label_paths))  # 打包成(img_path, label_path)元组列表
        # 打印该目录的测试样本数量
        print('{} raw samples for test from dir {}'.format(len(test_samples), i))  # 打印每个目录的测试样本数
        # 将该目录的测试样本添加到返回列表
        ret_test_samples.extend(test_samples)  # 添加到总测试样本列表

    # 打印总测试样本数量
    print('Total {} raw samples for test'.format(len(ret_test_samples)))  # 打印总测试样本数
    # 返回测试样本列表
    return ret_test_samples  # 返回测试样本路径列表

class TestRawDataset(Dataset):  # 测试集原始样本数据集类
    def __init__(self, raw_samples, img_size):
        self.raw_samples = raw_samples

        if img_size == (640, 480):
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(size=(img_size[1], img_size[0])),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        img_path, label_path = self.raw_samples[idx]

        img = self.img_transform(Image.open(img_path))
        label = load_xy_label(label_path)
        label = torch.from_numpy(label)

        return img, label, img_path, label_path
def prepare_loaders():
    test_raw_samples = get_test_raw_samples(
        img_dir_list=img_dir_list,
        label_dir_list=label_dir_list,
        train_size_list=train_size_list,
        dev_size_list=dev_size_list,
        test_size_list=test_size_list,
        random_seed=random_seed,
        img_size=img_size,
    )
    test_set = TestRawDataset(test_raw_samples, img_size=img_size)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader

def main():  # 主函数，执行推理流程
    test_loader = prepare_loaders()  # 准备测试数据加载器
    
    # 将基础图像路径转换为张量
    img_base_tensor = transforms.Compose([  # 创建图像变换管道
        transforms.Resize(size=(img_size[1], img_size[0])),  # 调整图像尺寸为指定大小
        transforms.ToTensor(),  # 将PIL Image转换为PyTorch张量
    ])(Image.open(img_base))  # 打开基础图像并应用变换
    
    # 将基础标签文件路径转换为标签
    label_base = load_xy_label(label_base_path)  # 加载基础标签数据
    label_base_tensor = torch.from_numpy(label_base).float()  # 将标签转换为PyTorch张量，确保为float32类型
    
    start = torch.cuda.Event(enable_timing=True)  # 创建CUDA计时开始事件，用于精确计时
    end = torch.cuda.Event(enable_timing=True)  # 创建CUDA计时结束事件，用于精确计时

    device = torch.device('cuda:{}'.format(GPU_IDS[0]))  # 设置计算设备为指定的GPU设备

    # 创建模型实例
    model = VSNet(num_classes=num_classes)  # 创建VSNet模型实例，输出维度为num_classes
    
    # 加载模型状态字典
    if os.path.exists(best_model_path):  # 检查最佳模型文件是否存在
        model_state_dict = torch.load(best_model_path, map_location=device)  # 加载最佳模型状态字典到指定设备
    model.load_state_dict(model_state_dict)  # 将加载的状态字典应用到模型实例

    # 如果需要多GPU评估
    if len(GPU_IDS) > 1:  # 检查是否使用多个GPU
        model = nn.DataParallel(model, device_ids=GPU_IDS)  # 使用DataParallel包装模型，支持多GPU并行计算

    model.eval()  # 设置模型为评估模式，关闭dropout和batch normalization等训练时的特性
    model.to(device)  # 将模型移动到指定的GPU设备
    
    # 将基础图像和标签移动到GPU
    img_base_tensor = img_base_tensor.to(device)  # 将基础图像张量移动到GPU设备
    label_base_tensor = label_base_tensor.to(device)  # 将基础标签张量移动到GPU设备
    
    # 添加batch维度
    img_base_tensor = img_base_tensor.unsqueeze(0)  # 在第0维添加维度，将[3, H, W]变为[1, 3, H, W]
    label_base_tensor = label_base_tensor.unsqueeze(0)  # 在第0维添加维度，将[2]变为[1, 2]
    
    # 初始化误差统计
    all_errors = []  # 初始化列表，用于保存所有样本的误差
    all_outputs = []  # 初始化列表，用于保存所有样本的模型输出
    all_labels = []  # 初始化列表，用于保存所有样本的真实标签
    all_img_paths = []  # 初始化列表，用于保存所有样本的图像路径
    all_label_paths = []  # 初始化列表，用于保存所有样本的标签文件路径
    
    # 创建结果保存目录
    os.makedirs(inference_results_dir, exist_ok=True)  # 创建结果保存目录，如果已存在则不报错
    results_file = inference_results_dir + '/test_errors.txt'  # 设置结果文件路径
    
    print('='*60)  # 打印分隔线
    print('Starting Inference')  # 打印开始推理的提示信息
    print('='*60)  # 打印分隔线
    print('Base image: {}'.format(img_base))  # 打印基础图像路径
    print('Base label: {}'.format(label_base))  # 打印基础标签值
    print('Test set size: {} samples'.format(len(test_loader)))  # 打印测试集样本数量
    print('='*60)  # 打印分隔线
    
    start_time = time.time()  # 记录推理开始时间，用于计算总推理时间
    
    # 遍历测试集
    with torch.no_grad():  # 禁用梯度计算，减少内存使用，加快推理速度
        for idx, (img_test, label_test, img_path, label_path) in enumerate(test_loader):  # 遍历测试数据加载器
            # 将测试图像和标签移动到GPU
            img_test = img_test.to(device)  # 将测试图像张量移动到GPU设备
            label_test = label_test.to(device)  # 将测试标签张量移动到GPU设备
            
            # 计算测试图像与基础图像之间的相对位姿
            output = model(img_test, img_base_tensor)  # 模型推理，输入测试图像和基础图像，输出相对位姿
            
            # 计算测试标签与基础标签之间的相对标签
            relative_label = label_test - label_base_tensor  # 计算相对标签，即测试标签减去基础标签
            
            # 将输出和标签移到CPU
            output_cpu = output.cpu().numpy()[0]  # 将模型输出移到CPU并转换为NumPy数组，去掉batch维度
            relative_label_cpu = relative_label.cpu().numpy()[0]  # 将相对标签移到CPU并转换为NumPy数组，去掉batch维度
            
            # 计算误差
            error = np.abs(output_cpu - relative_label_cpu)  # 计算预测值与真实值之间的绝对误差
            
            # 保存结果
            all_errors.append(error)  # 将当前样本的误差添加到误差列表
            all_outputs.append(output_cpu)  # 将当前样本的模型输出添加到输出列表
            all_labels.append(relative_label_cpu)  # 将当前样本的真实标签添加到标签列表
            # 提取并保存路径信息
            cur_img_path = img_path[0] if isinstance(img_path, (list, tuple)) else str(img_path)
            cur_label_path = label_path[0] if isinstance(label_path, (list, tuple)) else str(label_path)
            all_img_paths.append(cur_img_path)  # 保存图像路径
            all_label_paths.append(cur_label_path)  # 保存标签文件路径
            
            # 打印当前样本的结果
            print('Sample {}/{}:'.format(idx + 1, len(test_loader)))  # 打印当前样本的序号和总数
            print('  Image path: {}'.format(img_path[0]))  # 打印当前样本的图像路径
            print('  Predicted: x={:.2f}mm, y={:.2f}mm'.format(output_cpu[0], output_cpu[1]))  # 打印模型预测的x和y坐标
            print('  Ground truth: x={:.2f}mm, y={:.2f}mm'.format(relative_label_cpu[0], relative_label_cpu[1]))  # 打印真实的x和y坐标
            print('  Error: x={:.2f}mm, y={:.2f}mm'.format(error[0], error[1]))  # 打印预测误差的x和y分量
            print()  # 打印空行，分隔不同样本的输出
    
    # 计算总时间
    total_time = time.time() - start_time  # 计算总推理时间，当前时间减去开始时间
    avg_time_per_sample = total_time / len(test_loader)  # 计算每个样本的平均推理时间
    
    # 计算平均误差
    all_errors = np.array(all_errors)  # 将误差列表转换为NumPy数组，便于统计计算
    average_error = np.mean(all_errors, axis=0)  # 计算所有样本的平均误差，axis=0表示按列计算
    std_error = np.std(all_errors, axis=0)  # 计算所有样本的误差标准差，衡量误差的离散程度
    max_error = np.max(all_errors, axis=0)  # 计算所有样本的最大误差
    min_error = np.min(all_errors, axis=0)  # 计算所有样本的最小误差
    
    # 保存结果到文件
    with open(results_file, 'w') as f:  # 打开结果文件，'w'表示写入模式
        f.write('VSNet Inference Results\n')  # 写入标题
        f.write('='*60 + '\n')  # 写入分隔线
        f.write('Model: {}\n'.format(best_model_path))  # 写入模型路径
        f.write('Base image: {}\n'.format(img_base))  # 写入基础图像路径
        f.write('Base label: x={:.2f}mm, y={:.2f}mm\n'.format(label_base[0], label_base[1]))  # 写入基础标签值
        f.write('\n')  # 写入空行
        f.write('Test set size: {} samples\n'.format(len(test_loader)))  # 写入测试集样本数量
        f.write('\n')  # 写入空行
        f.write('Time Statistics:\n')  # 写入时间统计标题
        f.write('  Total time: {:.2f}s\n'.format(total_time))  # 写入总推理时间
        f.write('  Average time per sample: {:.4f}s\n'.format(avg_time_per_sample))  # 写入平均推理时间
        f.write('\n')  # 写入空行
        f.write('Error Statistics:\n')  # 写入误差统计标题
        f.write('  Average error: x={:.2f}mm, y={:.2f}mm\n'.format(average_error[0], average_error[1]))  # 写入平均误差
        f.write('  Standard deviation: x={:.2f}mm, y={:.2f}mm\n'.format(std_error[0], std_error[1]))  # 写入误差标准差
        f.write('  Maximum error: x={:.2f}mm, y={:.2f}mm\n'.format(max_error[0], max_error[1]))  # 写入最大误差
        f.write('  Minimum error: x={:.2f}mm, y={:.2f}mm\n'.format(min_error[0], min_error[1]))  # 写入最小误差
        f.write('\n')  # 写入空行
        f.write('Sample Details:\n')  # 写入样本详情标题
        f.write('='*60 + '\n')  # 写入分隔线
        for idx in range(len(all_errors)):  # 遍历所有样本的结果
            f.write('Sample {}/{}:\n'.format(idx + 1, len(all_errors)))  # 写入样本序号
            #f.write('  Image path: {}\n'.format(all_img_paths[idx]))  # 写入图像路径
            #f.write('  Label path: {}\n'.format(all_label_paths[idx]))  # 写入标签路径
            #f.write('  Predicted: x={:.2f}mm, y={:.2f}mm\n'.format(all_outputs[idx][0], all_outputs[idx][1]))  # 写入预测值
            #f.write('  Ground truth: x={:.2f}mm, y={:.2f}mm\n'.format(all_labels[idx][0], all_labels[idx][1]))  # 写入真实值
            f.write('  Error: x={:.2f}mm, y={:.2f}mm\n'.format(all_errors[idx][0], all_errors[idx][1]))  # 写入误差值
            f.write('\n')  # 写入空行
    
    # 打印统计结果
    print('='*60)  # 打印分隔线
    print('Inference Results')  # 打印推理结果标题
    print('='*60)  # 打印分隔线
    print('Test set size: {} samples'.format(len(test_loader)))  # 打印测试集样本数量
    print('Total time: {:.2f}s'.format(total_time))  # 打印总推理时间
    print('Average time per sample: {:.4f}s'.format(avg_time_per_sample))  # 打印平均推理时间
    print()  # 打印空行
    print('Error Statistics:')  # 打印误差统计标题
    print('  Average error: x={:.2f}mm, y={:.2f}mm'.format(average_error[0], average_error[1]))  # 打印平均误差
    print('  Standard deviation: x={:.2f}mm, y={:.2f}mm'.format(std_error[0], std_error[1]))  # 打印误差标准差
    print('  Maximum error: x={:.2f}mm, y={:.2f}mm'.format(max_error[0], max_error[1]))  # 打印最大误差
    print('  Minimum error: x={:.2f}mm, y={:.2f}mm'.format(min_error[0], min_error[1]))  # 打印最小误差
    print('='*60)  # 打印分隔线
    print('Results saved to: {}'.format(results_file))  # 打印结果文件保存路径
    print('Inference completed successfully!')  # 打印推理成功完成的提示信息

if __name__ == '__main__':  # 如果作为主程序运行
    main()  # 执行主函数