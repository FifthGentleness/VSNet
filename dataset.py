import os  # 导入操作系统相关功能的模块
import glob  # 导入用于文件路径匹配的模块
from PIL import Image  # 从PIL库导入Image模块，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch深度学习框架
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器类
from torchvision import transforms  # 导入PyTorch的图像变换模块
from utils import get_files, get_stem  # 从自定义工具模块导入辅助函数


def load_xy_label(label_path):
    """读取标签文件中的二维数值标签。"""
    label = np.loadtxt(label_path, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32).reshape(-1)
    assert label.size >= 2, 'Label file {} must contain at least two values'.format(label_path)
    return label[:2]


def generate_set_paths(img_paths, label_paths, aug_factor=None, limits=None, weights=[1/2, 1/2]):  # 生成数据集路径对的函数
    assert len(img_paths) == len(label_paths), 'Unequal number of image and label paths: {} vs {}'.format(
        len(img_paths), len(label_paths))  # 断言确保图像和标签路径数量相等

    set_paths = []  # 初始化路径对列表
    for a in range(len(img_paths)):  # 遍历所有图像路径

        if aug_factor is not None:  # 如果指定了增强因子
            assert aug_factor <= 1.0, 'aug_factor must not be larger than 1'  # 确保增强因子不大于1
            b_idx_list_len = int(len(img_paths) * aug_factor)  # 计算增强样本数量
            b_idx_list = np.random.permutation(len(img_paths))  # 随机排列索引
            b_idx_list = b_idx_list[:b_idx_list_len]  # 只选择部分索引
        else:
            b_idx_list = np.arange(len(img_paths))  # 使用所有索引

        for b in b_idx_list:  # 遍历选中的索引
            img_a_path = img_paths[a]  # 第一张图像路径
            img_b_path = img_paths[b]  # 第二张图像路径
            label_a_path = label_paths[a]  # 第一个标签路径
            label_b_path = label_paths[b]  # 第二个标签路径

            assert get_stem(img_a_path) == get_stem(label_a_path), \
                '{} and {} should have the same stem!'.format(img_a_path, label_a_path)  # 确保图像和标签文件名相同
            assert get_stem(img_b_path) == get_stem(label_b_path), \
                '{} and {} should have the same stem!'.format(img_b_path, label_b_path)  # 确保图像和标签文件名相同

            if limits is not None:  # 如果指定了偏差限制
                label_a = load_xy_label(label_a_path)
                label_b = load_xy_label(label_b_path)
                label = label_a - label_b

                weighted_deviation = weights[0] * np.abs(label[0]) + weights[1] * np.abs(label[1])
                if weighted_deviation > limits:  # 如果加权偏差超过限制
                    continue  # skip this example - 跳过这个样本

            set_paths.append((img_a_path, img_b_path, label_a_path, label_b_path))  # 添加路径对到列表

    return set_paths  # 返回路径对列表


def split_sets(img_dir_list, label_dir_list,  # 分割数据集的函数
               train_size_list=[600], dev_size_list=[200], test_size_list=[200],  # 各数据集大小列表
               random_seed=0, img_ext='.png', label_ext='.txt',  # 随机种子和文件扩展名
               aug_factor=None, limits=None, weights=[1/2, 1/2]):  # 增强因子、限制和权重
    """
    将数据集分割为训练集、验证集和测试集，并生成图像对路径
    
    参数:
        img_dir_list: 图像目录列表，每个目录包含一个数据集的图像
        label_dir_list: 标签目录列表，每个目录包含一个数据集的标签文件
        train_size_list: 训练集大小列表，每个元素对应一个数据集的训练样本数
        dev_size_list: 验证集大小列表，每个元素对应一个数据集的验证样本数
        test_size_list: 测试集大小列表，每个元素对应一个数据集的测试样本数
        random_seed: 随机种子，用于确保数据分割的可复现性
        img_ext: 图像文件扩展名，默认为'.png'
        label_ext: 标签文件扩展名，默认为'.txt'
        aug_factor: 数据增强因子，范围[0,1]，表示增强样本占总样本的比例
                   None表示不进行数据增强
        limits: 偏差限制，用于过滤偏差过大的样本
                None表示不限制
        weights: 损失权重列表，[平移权重, 旋转权重]，默认为[0.5, 0.5]
    
    返回:
        ret_train_paths: 训练集路径对列表，每个元素为(img_a_path, img_b_path, label_a_path, label_b_path)
        ret_dev_paths: 验证集路径对列表，格式同上
        ret_test_paths: 测试集路径对列表，格式同上
    """

    assert isinstance(img_dir_list, list), 'img_dir_list must be a list!'  # 确保输入是列表
    assert isinstance(label_dir_list, list), 'label_dir_list must be a list!'  # 确保输入是列表
    assert isinstance(train_size_list, list), 'train_size_list must be a list!'  # 确保输入是列表
    assert isinstance(dev_size_list, list), 'dev_size_list must be a list!'  # 确保输入是列表
    assert isinstance(test_size_list, list), 'test_size_list must be a list!'  # 确保输入是列表

    assert len(img_dir_list) == len(label_dir_list) == len(train_size_list) == len(dev_size_list) == len(test_size_list), 'lists must have equal lengths!'  # 确保所有列表长度相等

    ret_train_paths = []  # 初始化训练集路径列表
    ret_dev_paths = []  # 初始化验证集路径列表
    ret_test_paths = []  # 初始化测试集路径列表
    for i in range(len(img_dir_list)):  # 遍历所有目录

        img_dir = img_dir_list[i]  # 图像目录
        label_dir = label_dir_list[i]  # 标签目录
        train_size = train_size_list[i]  # 训练集大小
        dev_size = dev_size_list[i]  # 验证集大小
        test_size = test_size_list[i]  # 测试集大小

        img_paths = sorted(get_files(img_dir, img_ext))  # 获取排序后的图像文件路径
        label_paths = sorted(get_files(label_dir, label_ext))  # 获取排序后的标签文件路径

        assert len(img_paths) == len(label_paths), 'Unequal number of images and labels found!'  # 确保图像和标签数量相等
        assert len(img_paths) >= train_size + dev_size + test_size, 'Not enough images and labels are found'  # 确保有足够的样本

        np.random.seed(random_seed)  # 设置随机种子
        perm_list = np.random.permutation(len(img_paths))  # 随机排列索引
        img_paths = np.array(img_paths)  # 转换为NumPy数组
        label_paths = np.array(label_paths)  # 转换为NumPy数组

        train_img_paths = img_paths[perm_list[0:train_size]].tolist()  # 训练集图像路径
        train_label_paths = label_paths[perm_list[0:train_size]].tolist()  # 训练集标签路径
        dev_img_paths = img_paths[perm_list[train_size:train_size + dev_size]].tolist()  # 验证集图像路径
        dev_label_paths = label_paths[perm_list[train_size:train_size + dev_size]].tolist()  # 验证集标签路径
        test_img_paths = img_paths[perm_list[train_size + dev_size:train_size + dev_size + test_size]].tolist()  # 测试集图像路径
        test_label_paths = label_paths[perm_list[train_size + dev_size:train_size + dev_size + test_size]].tolist()  # 测试集标签路径

        train_paths = generate_set_paths(train_img_paths, train_label_paths, aug_factor=aug_factor, limits=limits, weights=weights)  # 生成训练集路径对
        print('{} samples for train from dir {}'.format(len(train_paths), i))  # 打印训练集样本数量
        ret_train_paths.extend(train_paths)  # 添加到返回列表

        dev_paths = generate_set_paths(dev_img_paths, dev_label_paths, aug_factor=aug_factor, limits=limits, weights=weights)  # 生成验证集路径对
        print('{} samples for dev from dir {}'.format(len(dev_paths), i))  # 打印验证集样本数量
        ret_dev_paths.extend(dev_paths)  # 添加到返回列表

        test_paths = generate_set_paths(test_img_paths, test_label_paths, aug_factor=aug_factor, limits=limits, weights=weights)  # 生成测试集路径对
        print('{} samples for test from dir {}'.format(len(test_paths), i))  # 打印测试集样本数量
        ret_test_paths.extend(test_paths)  # 添加到返回列表

    print('Total {} samples for train'.format(len(ret_train_paths)))  # 打印总训练样本数
    print('Total {} samples for dev'.format(len(ret_dev_paths)))  # 打印总验证样本数
    print('Total {} samples for test'.format(len(ret_test_paths)))  # 打印总测试样本数

    return ret_train_paths, ret_dev_paths, ret_test_paths  # 返回所有数据集路径

class VSDataset(Dataset):  # 视觉里程计数据集类

    def __init__(self, set_paths, img_size, return_path=False):  # 初始化函数
        self.set_paths = set_paths  # 存储路径对
        self.return_path = return_path  # 是否返回路径标志

        if img_size == (640, 480):  # 如果是原始图像尺寸
            # print('No resizing needed.')
            self.img_transform = transforms.Compose([  # 图像变换管道
                transforms.ToTensor(),  # 转换为张量
            ])
        else:  # 如果需要调整尺寸
            self.img_transform = transforms.Compose([  # 图像变换管道
                transforms.Resize(size=(img_size[1], img_size[0])), # TODO,bug: here should be (h,w)! - 调整图像尺寸
                transforms.ToTensor(),  # 转换为张量
            ])

    def __len__(self):  # 返回数据集大小
        return len(self.set_paths)  # 返回路径对数量

    def __getitem__(self, idx):  # 获取单个数据项
        img_a_path, img_b_path, label_a_path, label_b_path = self.set_paths[idx]  # 获取路径对

        img_a = Image.open(img_a_path)  # 打开第一张图像
        img_b = Image.open(img_b_path)  # 打开第二张图像
        label_a = load_xy_label(label_a_path)
        label_b = load_xy_label(label_b_path)
        label = label_a - label_b

        img_a = self.img_transform(img_a)  # 应用图像变换
        img_b = self.img_transform(img_b)  # 应用图像变换

        label = torch.from_numpy(label)  # 将标签转换为张量
        if self.return_path:  # 如果需要返回路径
            return img_a, img_b, label, img_a_path, img_b_path, label_a_path, label_b_path  # 返回图像、标签和路径
        else:
            return img_a, img_b, label  # 只返回图像和标签