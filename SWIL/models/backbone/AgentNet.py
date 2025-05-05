import torch.nn as nn
import torch
from einops import rearrange
import math
from torch_geometric.nn import GCNConv
import numpy as np
import scipy as sp
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))



import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

class AgentAttention(nn.Module):  # 定义一个名为AgentAttention的类，继承自PyTorch的nn.Module类，表示这是一个神经网络模块。
    # 定义类的构造函数，接收以下参数：
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., shift_size=0, agent_num=49, **kwargs):
        super().__init__()  # 调用父类的构造函数，进行必要的初始化。
        self.dim = dim  # 输入的维度。
        self.idf = nn.Identity()  # 创建一个恒等映射，不做任何改变。
        self.num_heads = num_heads  # 多头注意力的头数。
        head_dim = dim // num_heads  # 每个头的维度。
        self.scale = head_dim ** -0.5  # 注意力的缩放因子。
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 定义一个线性层，用于产生query、key和value。
        self.attn_drop = nn.Dropout(attn_drop)  # 定义注意力权重的dropout层。
        self.proj = nn.Linear(dim, dim)  # 定义一个线性层，用于投影。
        self.proj_drop = nn.Dropout(proj_drop)  # 定义输出投影的dropout层。
        self.softmax = nn.Softmax(dim=-1)  # 定义softmax函数。
        self.shift_size = shift_size  # 定义窗口的移动大小。
        self.agent_num = agent_num  # 定义agent的数量。
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1,
                             groups=dim)  # 定义一个2D卷积层。
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))  # 定义一个参数，表示attention的偏置。
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        # 使用截断正态分布初始化an_bias参数，其中std表示标准差，这里设置为0.02
        trunc_normal_(self.an_bias, std=.02)
        # 使用截断正态分布初始化na_bias参数，同样设置标准差为0.02
        trunc_normal_(self.na_bias, std=.02)
        # 计算pool_size的值，它是agent_num的平方根的整数部分
        pool_size = int(agent_num ** 0.5)
        # 定义一个自适应平均池化层，其输出大小为(pool_size, pool_size)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x):
        # 获取输入x经过idf层的输出，即shortcut
        shortcut = self.idf(x)

        # 获取shortcut的形状，并分别提取高和宽
        h, w = shortcut.shape[2], shortcut.shape[3]

        # 将输入x展平，并交换维度2和1的位置
        x = x.flatten(2).transpose(1, 2)

        # 获取展平后x的形状，并分别提取batch_size、序列长度和通道数
        b, n, c = x.shape

        # 定义多头注意力中的头数
        num_heads = self.num_heads

        # 计算每个头的维度
        head_dim = c // num_heads

        # 对输入x进行qkv变换，并重新整形和转置
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)

        # 将qkv的结果分为q、k和v三个部分
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 输出q、k、v的形状信息
        # q, k, v: b, n, c

        # 对q进行池化操作，得到agent_tokens，并对形状进行转置和重新整形
        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)

        # 对q、k、v进行形状的重新整形和转置，以便进行多头注意力计算
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        # 对agent_tokens进行形状的重新整形和转置，以便与q、k、v进行多头注意力计算
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # 初始化ah_bias和aw_bias，这两个偏置用于位置编码
        ah_bias = torch.as_tensor(torch.zeros(1, num_heads, self.agent_num, h, 1).to(x.device).detach(), dtype=x.dtype)
        aw_bias = torch.as_tensor(torch.zeros(1, num_heads, self.agent_num, 1, w).to(x.device).detach(), dtype=x.dtype)

        # 对an_bias进行上采样，得到与输入x相同的大小
        position_bias1 = nn.functional.interpolate(self.an_bias, size=(h, w), mode='bilinear')
        # 对上采样后的位置偏置进行形状调整，使其与后面的计算相匹配
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # 将ah_bias和aw_bias拼接，并与上采样后的位置偏置相加
        position_bias2 = (ah_bias + aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        # 对agent_tokens进行softmax操作，并与key进行矩阵乘法，再加上位置偏置
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        # 应用dropout
        agent_attn = self.attn_drop(agent_attn)
        # 对value进行加权求和，得到agent_v
        agent_v = agent_attn @ v

        # 初始化ha_bias和wa_bias，这两个偏置用于另一个位置编码
        ha_bias = torch.as_tensor(torch.zeros(1, num_heads, h, 1, self.agent_num).to(x.device).detach(), dtype=x.dtype)
        wa_bias = torch.as_tensor(torch.zeros(1, num_heads, 1, w, self.agent_num).to(x.device).detach(), dtype=x.dtype)

        # 对na_bias进行上采样，得到与输入x相同的大小
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(h, w), mode='bilinear')
        # 对上采样后的位置偏置进行形状调整，使其与后面的计算相匹配
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        # 将ha_bias和wa_bias拼接，并与上采样后的位置偏置相加
        agent_bias2 = (ha_bias + wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        # 对q和agent_tokens进行softmax操作，并进行矩阵乘法，再加上位置偏置
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        # 应用dropout
        q_attn = self.attn_drop(q_attn)
        # 对value进行加权求和，得到x的一部分结果
        x = q_attn @ agent_v

        # 对x进行形状调整，使其与shortcut相匹配
        x = x.transpose(1, 2).reshape(b, n, c)
        # 对v进行形状调整，使其与shortcut相匹配
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # 将x与shortcut相加，并应用dwc层（可能是某种变换或池化层）
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        # 通过一个线性层proj进行变换
        x = self.proj(x)

        # 通过一个dropout层proj_drop进行dropout操作，以防止过拟合
        x = self.proj_drop(x)

        # 将x的形状重新调整为(b, h, w, c)，其中b是batch size，h和w是高和宽，c是通道数
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # 将shortcut与sigmoid函数的结果相乘，sigmoid函数可以将输入映射到0到1之间，这里可能是为了将输出限制在一个合理的范围内
        x = shortcut * torch.sigmoid(x)
        return x


    # 定义一个flops方法，该方法接受一个参数N，表示输入序列的长度
    def flops(self, N):
        # 初始化浮点运算次数的计数器为0
        flops = 0
        # 这行代码被注释掉了，它似乎是计算qkv矩阵的FLOPs，其中qkv是query、key和value矩阵
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        # 这行代码计算了qkv矩阵的FLOPs，其中N是序列长度，self.dim是维度大小，3是因为qkv有3个矩阵（query、key和value）
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # 这行代码计算了查询矩阵（q）和键矩阵（k）转置的乘法操作的FLOPs，其中self.num_heads是头数，N是序列长度，self.dim // self.num_heads是每个头的维度大小
        # x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # 这行代码计算了注意力权重矩阵（attn）和值矩阵（v）的乘法操作的FLOPs
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        # 这行代码计算了经过线性变换后的输出的FLOPs，其中N是序列长度，self.dim是维度大小
        return flops

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),

        )
        self.gat = GCNConv(channel // reduction, channel, bias=False)
        self.s = nn.Sigmoid()

    def create_edge_index(self, adj, device):
        adj = adj.cpu()
        ones = torch.ones_like(adj)
        zeros = torch.zeros_like(adj)
        edge_index = torch.where(adj > 0, ones, zeros)
        #
        edge_index_temp = sp.sparse.coo_matrix(edge_index.numpy())
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
        edge_index = torch.LongTensor(indices)
        # edge_weight
        edge_weight = []
        t = edge_index.numpy().tolist()
        for x, y in zip(t[0], t[1]):
            edge_weight.append(adj[x, y])
        edge_weight = torch.FloatTensor(edge_weight)
        edge_weight = edge_weight.unsqueeze(1)

        return edge_index.to(device), edge_weight.to(device)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = torch.unsqueeze(y, 1)
        s = torch.randn((y.shape[0], y.shape[1], c)).to(x.device)
        for k in range(y.shape[0]):
            feat = y[k, :, :]  # s, h
            # creat edge_index
            adj = torch.matmul(feat, feat.T)  # s * s
            adj = F.softmax(adj, dim=1)
            edge_index, edge_weight = self.create_edge_index(adj, x.device)
            feat = self.gat(feat, edge_index, edge_weight)
            s[k, :, :] = feat
        s = s[:, -1, :]
        y = torch.as_tensor(s.view(b, c, 1, 1), dtype=x.dtype)
        return x * y.expand_as(x)

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(AgentAttention(self.c, 4, True, 0.2, 0.2) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))




class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class AgnetNet(nn.Module):
    def __init__(self, depth=[2,2,6,2]):
        super().__init__()
        self.stage1 = nn.Sequential(
            Conv(3, 64, 3, 2),
            Conv(64, 128, 3, 2),
            C2f(128, 128, depth[0], True)
        )
        self.stage2 = nn.Sequential(
            Conv(128, 256, 3, 2),
            C2f(256, 256, depth[1], True)
        )
        self.stage3 = nn.Sequential(
            Conv(256, 512, 3, 2),
            C2f(512, 512, depth[2], True)
        )
        self.stage4 = nn.Sequential(
            Conv(512, 1024, 3, 2),
            C2f(1024, 1024, depth[3], True),
            SPPF(1024, 1024, 5)
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x1, x2, x3, x4


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 640, 640)
    image = torch.rand(*image_size).cuda()

    # Model
    model = AgnetNet(depth=[2,2,6,2]).cuda()
    print(model)
    out = model(image)
    print([tensor.shape for tensor in out])