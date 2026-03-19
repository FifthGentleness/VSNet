from imgaug import augmenters as iaa  # 导入图像增强库imgaug的增强器模块
import glob  # 导入glob模块，用于文件路径匹配
import os  # 导入os模块，用于操作系统相关功能
from PIL import Image  # 从PIL库导入Image模块，用于图像处理


seq = iaa.Sequential([  # 创建一个序列化的图像增强管道
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen) - 随机裁剪图像，每侧裁剪0到16像素
    iaa.Fliplr(0.5), # horizontally flip 50% of the images - 以50%的概率水平翻转图像
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0 - 对图像应用高斯模糊，sigma值在0到3.0之间随机
])

def main():  # 主函数，程序入口
    aug()  # 调用图像增强函数

def aug():  # 图像增强函数
    os.chdir('./six_dof_1cm5deg_text')  # 切换到指定目录，该目录包含待增强的图像
    for picture in range(sorted(glob.glob('*.png'))):  # 遍历目录中所有PNG图像文件（按名称排序）
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
    # 以上注释说明：'images'应该是一个4D numpy数组，形状为(N,高度,宽度,通道)
    # 或一个3D numpy数组列表，每个数组形状为(高度,宽度,通道)
    # 灰度图像必须具有形状(高度,宽度,1)
    # 所有图像必须是numpy的uint8类型，值范围应在0-255之间
        images = Image.open(picture)  # 使用PIL打开图像文件
        images_aug = seq.augment_images(images)  # done by the library - 使用预定义的增强管道对图像进行增强处理
        j = Image.fromarray(images_aug, mode='RGB')  # 将增强后的numpy数组转换回PIL图像对象，模式为RGB
        j.save(picture)  # 保存增强后的图像，覆盖原始文件