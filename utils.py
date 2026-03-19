import os  # 导入操作系统相关功能的模块
import sys  # 导入系统相关功能的模块
import glob  # 导入文件路径匹配模块
import numpy as np  # 导入NumPy库，用于数值计算
import transformations as tr  # 导入变换处理模块


def accuracy_thres_curve(error_list, thres_list):  # 计算精度-阈值曲线函数
    """
    计算误差在不同阈值下的精度
    
    参数:
        error_list: 误差列表
        thres_list: 阈值列表
    
    返回:
        accuracy_list: 精度列表，每个元素表示对应阈值下的精度
        thres_list: 阈值列表
    """
    num_error = len(error_list)  # 获取误差数量
    num_thres = len(thres_list)  # 获取阈值数量

    accuracy_list = [0] * num_thres  # 初始化精度列表

    error_list = sorted(error_list)  # 对误差列表进行排序

    error_idx = 0  # 初始化误差索引
    thres_idx = 0  # 初始化阈值索引

    # 遍历误差和阈值列表
    while error_idx < num_error and thres_idx < num_thres:
        if error_list[error_idx] <= thres_list[thres_idx]:  # 如果当前误差小于等于当前阈值
            accuracy_list[thres_idx] = error_idx + 1  # 更新当前阈值的精度
            error_idx += 1  # 移动到下一个误差
        else:
            accuracy_list[thres_idx] = error_idx  # 更新当前阈值的精度
            thres_idx += 1  # 移动到下一个阈值

    # 处理剩余的阈值
    while thres_idx < num_thres:
        accuracy_list[thres_idx] = error_idx + 1  # 所有误差都小于当前阈值
        thres_idx += 1  # 移动到下一个阈值

    # 计算精度百分比
    accuracy_list = list(np.array(accuracy_list) / num_error)

    return accuracy_list, thres_list  # 返回精度列表和阈值列表


def axis_angle_from_quat(q):  # 从四元数提取轴角表示函数
    """
    将四元数转换为轴角表示
    
    参数:
        q: 四元数，格式为[w, x, y, z]，其中w为标量部分
    
    返回:
        axis: 旋转轴，3D向量
        angle: 旋转角度，弧度
    """
    if q[0] > 1:  # 如果四元数的标量部分大于1，进行归一化
        norm = np.sqrt(np.sum(q ** 2))  # 计算四元数的模
        q /= norm  # 归一化四元数

    angle = 2 * np.arccos(q[0])  # 计算旋转角度

    s = np.sqrt(1 - q[0] ** 2)  # 计算旋转轴的缩放因子
    if s < 1e-4:  # 如果角度接近0，旋转轴任意
        axis = q[1:]  # 直接使用向量部分作为旋转轴
    else:
        axis = q[1:] / s  # 计算归一化的旋转轴

    return axis, angle  # 返回旋转轴和角度


def normalize_q(q):  # 归一化四元数函数
    """
    归一化四元数
    
    参数:
        q: 四元数
    
    返回:
        归一化后的四元数
    """
    q = np.array(q)  # 转换为NumPy数组
    assert q.shape == (4,)  # 确保四元数是4维向量
    norm = np.sqrt(np.sum(q ** 2))  # 计算四元数的模
    return q / norm  # 返回归一化后的四元数


def tf_to_quat(T):  # 将变换矩阵转换为平移和四元数函数
    """
    将4x4变换矩阵转换为平移向量和四元数
    
    参数:
        T: 变换矩阵，可以是3x4或4x4
    
    返回:
        7D向量，前3个元素为平移，后4个元素为四元数
    """
    if T.shape == (12,):  # 如果是12D向量
        T.shape = (3,4)  # 重塑为3x4矩阵
    if T.shape == (3,4):  # 如果是3x4矩阵
        T = np.concatenate((T, np.array([[0.0,0.0,0.0,1.0]])), axis=0)  # 添加最后一行，转换为4x4矩阵
    assert T.shape == (4, 4)  # 确保是4x4矩阵
    translation = tr.translation_from_matrix(T)  # 提取平移向量
    # quaternions = np.array(tr.quaternion_from_matrix(T[0:3, 0:3]))  # 可选：只从旋转部分提取四元数
    quaternions = np.array(tr.quaternion_from_matrix(T))  # 从整个矩阵提取四元数
    return np.concatenate((translation, quaternions), axis=0)  # 连接平移和四元数


def tf_to_dof(T):  # 将变换矩阵转换为平移和欧拉角函数
    """
    将4x4变换矩阵转换为平移向量和欧拉角
    
    参数:
        T: 变换矩阵，可以是3x4或4x4
    
    返回:
        6D向量，前3个元素为平移，后3个元素为欧拉角
    """
    if T.shape == (12,):  # 如果是12D向量
        T.shape = (3,4)  # 重塑为3x4矩阵
    if T.shape == (3,4):  # 如果是3x4矩阵
        T = np.concatenate((T, np.array([[0.0,0.0,0.0,1.0]])), axis=0)  # 添加最后一行，转换为4x4矩阵
    assert T.shape == (4,4), 'T.shape is {}, not (4,4)'.format(T.shape)  # 确保是4x4矩阵
    translation = tr.translation_from_matrix(T)  # 提取平移向量
    # euler_angles = np.array(tr.euler_from_matrix(T[0:3, 0:3]))  # 可选：只从旋转部分提取欧拉角
    euler_angles = np.array(tr.euler_from_matrix(T))  # 从整个矩阵提取欧拉角
    return np.concatenate((translation, euler_angles), axis=0)  # 连接平移和欧拉角


def dof_to_tf(x, y, z, roll, pitch, yaw):  # 将平移和欧拉角转换为变换矩阵函数
    """
    将平移向量和欧拉角转换为4x4变换矩阵
    
    参数:
        x, y, z: 平移分量
        roll, pitch, yaw: 欧拉角（度）
    
    返回:
        4x4变换矩阵
    """
    T = tr.euler_matrix(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw))  # 从欧拉角创建旋转矩阵
    T[:3, 3] = [x, y, z]  # 设置平移部分
    return T  # 返回变换矩阵


def get_stem(path):  # 获取文件名（不含扩展名）函数
    """
    从文件路径中提取文件名（不含扩展名）
    
    参数:
        path: 文件路径
    
    返回:
        文件名（不含扩展名）
    """
    basename = os.path.basename(path)  # 获取基本文件名
    stem, _ = os.path.splitext(basename)  # 分离文件名和扩展名
    return stem  # 返回文件名（不含扩展名）


def get_extension(path):  # 获取文件扩展名函数
    """
    从文件路径中提取文件扩展名
    
    参数:
        path: 文件路径
    
    返回:
        文件扩展名
    """
    basename = os.path.basename(path)  # 获取基本文件名
    _, ext = os.path.splitext(basename)  # 分离文件名和扩展名
    return ext  # 返回扩展名


def get_files(my_dir, ext):  # 获取目录中指定扩展名的文件列表函数
    """
    获取目录中所有指定扩展名的文件路径
    
    参数:
        my_dir: 目录路径
        ext: 文件扩展名
    
    返回:
        文件路径列表
    """
    if my_dir[-1] != '/':  # 如果目录路径不以斜杠结尾
        my_dir += '/'  # 添加斜杠
    assert os.path.isdir(my_dir), '{} is not a valid directory!'.format(my_dir)  # 确保目录存在

    if ext[0] != '.':  # 如果扩展名不以点开头
        ext = '.' + ext  # 添加点

    files = glob.glob(my_dir + '*' + ext)  # 获取所有匹配的文件
    return files  # 返回文件列表


def make_dir(my_dir):  # 创建目录函数
    """
    创建目录，如果目录已存在则不执行任何操作
    
    参数:
        my_dir: 目录路径
    """
    if os.path.isdir(my_dir):  # 如果目录已存在
        pass  # 不执行任何操作
    else:  # 如果目录不存在
        os.makedirs(my_dir)  # 创建目录
        print('Make directory {}'.format(my_dir))  # 打印创建信息


# Make directory if directory does not exist.
# Ask to remove content if directory exists.
def check_dir(my_dir):  # 检查并准备目录函数
    """
    检查目录是否存在，如果不存在则创建
    如果目录存在且不为空，询问用户是否删除内容
    
    参数:
        my_dir: 目录路径
    """
    if my_dir[-1] == '/':  # 如果目录路径以斜杠结尾
        my_dir = my_dir.rstrip('/')  # 移除末尾的斜杠

    make_dir(my_dir)  # 创建目录（如果不存在）

    files = os.listdir(my_dir)  # 获取目录中的文件列表
    if files:  # 如果目录不为空
        while True:  # 循环直到用户做出决定
            is_delete = input("Files found in " + my_dir + ". They will be removed if you continue. Continue? [ENTER/n]")  # 询问用户
            if is_delete == '':  # 如果用户按回车键
                for file in files:  # 遍历所有文件
                    os.remove(my_dir + '/' + file)  # 删除文件
                break  # 退出循环
            elif is_delete == 'n':  # 如果用户输入n
                sys.exit(-1)  # 退出程序