import torch
torch.manual_seed(2020)

# ======================================= example 1 =======================================
# torch.cat
flag = False
if flag:
    t = torch.ones((2, 3))
    t_0 = torch.cat([t, t], dim=0)
    t_1 = torch.cat([t, t, t], dim=1)
    print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))

# ======================================= example 2 =======================================
# torch.stack
flag = False
if flag:
    t = torch.ones((2, 3))
    t_stack = torch.stack([t, t, t], dim=0)  # 在第一个维度上拼接
    print("\nt_stack:{} shape:{}".format(t_stack, t_stack.shape))   # [3, 2, 3]
    t_stack = torch.stack([t, t], dim=0)  # 在第一个维度上拼接
    print("\nt_stack:{} shape:{}".format(t_stack, t_stack.shape))  # [2, 2, 3]

# ======================================= example 3 =======================================
# torch.chunk  将张量按维度dim进行平均切分
flag = False
if flag:
    a = torch.ones((2, 7))
    list_of_tensors = torch.chunk(a, dim=1, chunks=3)  # 在第二维度上切分三块
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))  # [2, 3]  [2, 3], [2, 1]

# ======================================= example 4 =======================================
# torch.split  切分
flag = False
if flag:
    t = torch.ones((2, 5))

    list_of_tensors = torch.split(t, [2, 1, 2], dim=1)
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))  # [2, 2] [2, 1]  [2, 2]

# ======================================= example 5 =======================================
# torch.index_select   在维度dim上，按index索引数据，返回拼接的张量
flag = False
if flag:
    t = torch.randint(0, 9, size=(3, 3))   # [0,9)之间均匀分布分整数
    idx = torch.tensor([0, 2], dtype=torch.long)  # float
    t_select = torch.index_select(t, dim=0, index=idx)
    print("t:\n{}\nt_select:\n{}".format(t, t_select))

# ======================================= example 6 =======================================
# torch.masked_select  按mask中的True进行索引
flag = False
if flag:
    t = torch.randint(0, 9, size=(3, 3))
    mask = t.le(5)  # 小于等于5的元素
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t, mask, t_select))

# ======================================= example 7 =======================================
# torch.reshape
flag = False
if flag:
    t = torch.randperm(8)   # 生成从0~n-1的随机排列
    t_reshape = torch.reshape(t, (-1, 2, 2))  # -1
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))    # [2, 2, 2]

    t[0] = 1024
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
    print("t.data 内存地址:{}".format(id(t.data)))
    print("t_reshape.data 内存地址:{}".format(id(t_reshape.data)))    # 内存地址相同

# ======================================= example 8 =======================================
# torch.transpose
flag = False
if flag:
    # torch.transpose
    t = torch.rand((2, 3, 4))    # 在[0,1)之间的均匀分布
    t_transpose = torch.transpose(t, dim0=1, dim1=2)    #  c*h*w --->  h*w*c
    print("t shape:{}\nt_transpose shape: {}".format(t.shape, t_transpose.shape))

# ======================================= example 9 =======================================
# torch.squeeze

# flag = True
flag = False

if flag:
    t = torch.rand((1, 2, 3, 1))
    t_sq = torch.squeeze(t)
    t_0 = torch.squeeze(t, dim=0)
    t_1 = torch.squeeze(t, dim=1)
    print(t.shape)
    print(t_sq.shape)
    print(t_0.shape)
    print(t_1.shape)

# ======================================= example 10 =======================================
# torch.add

flag = True
# flag = False

if flag:
    t_0 = torch.randn((3, 3))
    t_1 = torch.ones_like(t_0)
    t_add = torch.add(t_0, 10, t_1)

    print("t_0:\n{}\nt_1:\n{}\nt_add_10:\n{}".format(t_0, t_1, t_add))
