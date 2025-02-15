from scipy.signal import hilbert
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def threshold_adjacency_matrix(adj_matrix, percentage):

    adj_matrix=np.array(adj_matrix.cpu())
    # 计算保留的元素个数
    num_elements = adj_matrix.size
    num_to_keep = int(np.floor(num_elements * percentage / 100))
    # 展平矩阵并排序
    flat_matrix = adj_matrix.flatten()
    threshold_value = np.partition(flat_matrix, -num_to_keep)[-num_to_keep]
    # 将小于阈值的元素置为0
    thresholded_matrix = np.where(adj_matrix >= threshold_value, adj_matrix, 0)
    tadj=torch.tensor(thresholded_matrix,device='cuda:0')
    return tadj


class PLILayer(nn.Module):
    def __init__(self,threshold=0):
        super(PLILayer, self).__init__()
        self.threshold= threshold

    @staticmethod
    def compute_pli(x, y):
        thetax = torch.angle(torch.tensor(hilbert(x.cpu().numpy())).to('cuda:0'))
        thetay = torch.angle(torch.tensor(hilbert(y.cpu().numpy())).to('cuda:0'))
        thetadiff = thetax - thetay
        thetadiffmodify = torch.angle(torch.exp(1j * thetadiff))
        thetadiffmodify[(thetadiffmodify == np.pi) | (thetadiffmodify == -np.pi)] = 0
        return torch.abs(torch.mean(torch.sign(thetadiffmodify)))

    def forward(self, x):
        batch_size, channels, samples = x.shape
        PLI = torch.zeros((batch_size, channels, channels), dtype=torch.float32)
        for batch in range(batch_size):
            for i in range(channels):
                for j in range(channels):           #   这个地方ij算了两遍，时间浪费了一倍，可以减少
                    if i != j:
                        value=self.compute_pli(x[batch, i, :], x[batch, j, :])
                        if value>=self.threshold:
                            PLI[batch, i, j] = value
                        else:
                            PLI[batch, i, j] = 0
        return PLI


class PLILayer2(nn.Module):
    def __init__(self,threshold=0):
        super(PLILayer2, self).__init__()
        self.threshold= threshold

    @staticmethod
    def compute_pli(x, y):
        thetax = torch.angle(torch.tensor(hilbert(x.cpu().numpy())).to('cuda:0'))
        thetay = torch.angle(torch.tensor(hilbert(y.cpu().numpy())).to('cuda:0'))
        thetadiff = thetax - thetay
        thetadiffmodify = torch.angle(torch.exp(1j * thetadiff))
        thetadiffmodify[(thetadiffmodify == np.pi) | (thetadiffmodify == -np.pi)] = 0
        return torch.abs(torch.mean(torch.sign(thetadiffmodify)))

    def forward(self, x):
        batch_size, channels, samples = x.shape
        PLI = torch.zeros((batch_size, channels, channels), dtype=torch.float32)
        for batch in range(batch_size):
            for i in range(channels):
                for j in range(i):           #   这个地方ij算了两遍，时间浪费了一倍，可以减少
                    value=self.compute_pli(x[batch, i, :], x[batch, j, :])
                    if value>=self.threshold:
                        PLI[batch, i, j] = PLI[batch, j, i] = value

        return PLI



class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：H*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 初始化w

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        # init.kaiming_uniform_神经网络权重初始化，神经网络要优化一个非常复杂的非线性模型，而且基本没有全局最优解，
        # 初始化在其中扮演着非常重要的作用，尤其在没有BN等技术的早期，它直接影响模型能否收敛。

        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.matmul(input_feature, self.weight)
        output = torch.matmul(adjacency.to('cuda:0'), support.transpose(0,1)).transpose(0,1)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """
    def Convlayer(self,input_dim):
        return nn.Sequential(
            nn.Conv2d(input_dim,input_dim,(1,10)),
            nn.BatchNorm2d(input_dim),
            nn.ELU(),
            nn.MaxPool2d((1,10),stride=(1,10)),
            nn.Dropout(p=0.25)
        )
    def Conlayer2(self):
        return nn.Sequential(
            nn.Conv2d(9,288,(204,1)),
            nn.BatchNorm2d(288),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )

    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.Conv = self.Convlayer(input_dim)
        self.gcn1 = GraphConvolution(249, 64)
        self.gcn2 = GraphConvolution(64, 5)
        self.Conv2=self.Conlayer2()



    def forward(self, feature,adjacency):
        feature=self.Conv(feature)
        adjacency=threshold_adjacency_matrix(adjacency,85)
        hs=self.gcn1(adjacency, feature)
        h = F.relu(hs)
        logits = self.gcn2(adjacency, h)
        logits=self.Conv2(logits)
        return logits





