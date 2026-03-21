import torch  # 导入PyTorch深度学习框架
import torchvision  # 导入PyTorch的计算机视觉模块
import os  # 导入操作系统相关功能的模块
import numpy as np  # 导入NumPy库，用于数值计算
from torchvision import transforms  # 导入PyTorch的图像变换模块
from PIL import Image  # 从PIL库导入Image模块，用于图像处理


class VSNet(object):  # VSNet视觉里程计网络类
    def __init__(self, model, use_gpu= True):  # 初始化函数，参数为模型路径和是否使用GPU

        assert os.path.isfile(model), 'Model does not exists'  # 断言确保模型文件存在
        if torch.cuda.is_available() and not use_gpu:  # 如果有CUDA支持但未使用GPU
            print('Your computer do have cuda support, please use it')  # 提示用户使用GPU

        if use_gpu:  # 如果选择使用GPU
            assert torch.cuda.is_available(), 'No cuda support'  # 断言确保有CUDA支持
            self.device = torch.device('cuda')  # 设置设备为CUDA

        self.model = torch.load(model, map_location=self.device)  # 加载模型到指定设备
        self.model = self.model.module  # 获取模型模块（可能用于处理DataParallel包装的模型）
        self.model.eval()  # 设置模型为评估模式
        self.img_transform = transforms.Compose([transforms.ToTensor(),])  # 定义图像变换管道，只转换为张量

    def infer(self,img_a, img_b):  # 推理函数，输入为两张图像路径
        assert os.path.isfile(img_a), 'The first image does not exist'  # 断言确保第一张图像存在
        assert os.path.isfile(img_b), 'The second image does not exist'  # 断言确保第二张图像存在

        img_a = self.img_transform(Image.open(img_a))  # 加载并变换第一张图像
        img_b = self.img_transform(Image.open(img_b))  # 加载并变换第二张图像
        img_a = img_a.unsqueeze(0)  # 增加批次维度
        img_b = img_b.unsqueeze(0)  # 增加批次维度

        img_a = img_a.to(self.device)  # 将第一张图像移动到指定设备
        img_b = img_b.to(self.device)  # 将第二张图像移动到指定设备
        output = self.model(img_a, img_b)  # 模型推理，输出相对位姿
        output = output.cpu().detach().numpy()  # 将输出移回CPU并转换为NumPy数组

        return np.array(output[0])  # 返回第一个（也是唯一一个）样本的输出


def load_xy_label(label_path):
    """读取标签文件中的二维数值标签。"""
    label = np.loadtxt(label_path, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32).reshape(-1)
    assert label.size >= 2, 'Label file {} must contain at least two values'.format(label_path)
    return label[:2]


def main():  # 主函数
    model = './model.pth'  # 模型文件路径
    img_a = './test_img/image_a.png'  # 第一张测试图像路径
    img_b = './test_img/image_b.png'  # 第二张测试图像路径
    label_a_path = './six_dof_1cm5deg/label/607.txt'  # 第一个标签文件路径
    label_b_path = './six_dof_1cm5deg/label/100.txt'  # 第二个标签文件路径
    start = torch.cuda.Event(enable_timing=True)  # 创建CUDA计时开始事件
    end = torch.cuda.Event(enable_timing=True)  # 创建CUDA计时结束事件


    label_a = load_xy_label(label_a_path)
    label_b = load_xy_label(label_b_path)
    label = label_a - label_b

    net = VSNet(model)  # 创建VSNet网络实例

    start.record()  # 开始计时

    # Waits for everything to finish running
    output = net.infer(img_a, img_b)  # 进行推理
    end.record()  # 结束计时
    torch.cuda.synchronize()  # 等待所有CUDA操作完成
    print(output)  # 打印推理结果
    print(start.elapsed_time(end))  # 打印推理耗时

if __name__ == '__main__':  # 如果作为主程序运行
    main()  # 执行主函数