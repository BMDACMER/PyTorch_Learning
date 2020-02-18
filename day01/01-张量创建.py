import torch
import numpy as np
torch.manual_seed(2020)   # 设置随机种子，便于结果复现结果

# --------------exer1------------------------
# 通过torch.tensor创建张量
flag = False
if flag:
    arr = np.ones([3, 3])   # 这里用小括号也可以 arr = np.ones((3, 3))
    print("ndarray的数据类型：", arr.dtype)

    t = torch.tensor(arr, device='cuda')  # 使用gpu
    # t = torch.tensor(arr)               # 使用cpu
    print(t)

# --------------------exer2--------------------
# 从torch.from_numpy创建的tensor与原ndarray共享内存，当修改其中一个的数据，另一个也会被改动
flag = False
if flag:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    t = torch.from_numpy(arr)
    print("numpy array: \n", arr)
    print("tensor : \n", t)

    print("\n修改arr")
    arr[0, 0] = 0
    print("numpy array: ", arr)
    print("tensor : ", t)

# ===============================  exmaple 3 ===============================
# 通过torch.zeros创建张量   验证共享内存
flag = False
if flag:
    out_t = torch.tensor([1])

    t = torch.zeros((3, 3), out=out_t)
    print(t, '\n', out_t)
    print(id(t), id(out_t), id(t) == id(out_t))   # true

# ===============================  exmaple 4 ===============================
# 通过torch.full创建全1张量
flag = False
if flag:
    t = torch.full((3, 3), 1)  # （3，3）元素维1的矩阵
    print(t)

# ===============================  exmaple 5 ===============================
# 通过torch.arange创建等差数列张量
flag = False
if flag:
    t = torch.arange(2, 10, 2)  # 数值区间为[start, end, step)
    print(t)

# ===============================  exmaple 6 ===============================
# 通过torch.linspace创建均分数列张量
flag = False
if flag:
    t = torch.linspace(2, 10, 5)  # 区间为[start, end]
    print(t)   # tensor([ 2.,  4.,  6.,  8., 10.])

# ===============================  exmaple 7 ===============================
# 通过torch.normal创建正态分布张量
flag = True
if flag:
    # mean：张量 std: 张量
    # mean = torch.arange(1, 5, dtype=torch.float)
    # std = torch.arange(1, 5, dtype=torch.float)
    # t_normal = torch.normal(mean, std)
    # print("mean:{}\nstd:{}".format(mean, std))
    # print(t_normal)

    # mean：标量 std: 标量
    # t_normal = torch.normal(0., 1., size=(4,))
    # print(t_normal)

    # mean：张量 std: 标量
    mean = torch.arange(1, 5, dtype=torch.float)
    std = 1
    t_normal = torch.normal(mean, std)  # 一张量一标量
    print("mean:{}\nstd:{}".format(mean, std))
    print("t_normal:",t_normal)

