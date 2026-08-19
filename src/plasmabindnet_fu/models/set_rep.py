import torch
from torch.nn import Module, Parameter, Linear, BatchNorm1d, ReLU, Linear
import torch.nn.functional as F


class SetRep(Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        d: int,
        n_out_channels: int = 32,
    ):
        super(SetRep, self).__init__()

        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements
        self.d = d
        self.n_out_channels = n_out_channels

        self.Wc = Parameter(
            torch.FloatTensor(self.d, self.n_hidden_sets * self.n_elements)
        ) #shape is (d, n_hidden_sets*n_elements) 133, 64*8

        self.bn = BatchNorm1d(self.n_hidden_sets) #64
        self.fc1 = Linear(self.n_hidden_sets, self.n_out_channels) #64, 32
        self.relu = ReLU()

        # Init weights
        self.Wc.data.normal_()
    
    
    def forward(self, X):
        t = self.relu(torch.matmul(X, self.Wc)) # (batch_size, max_cliques, d) (64,29,300) #这里对应应该是max_cliques, 而不是原始代码里的max_atoms,原始代码是按照节点汇总，这里按照簇汇总
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets) # (batch_size, max_cliques, n_elements, n_hidden_sets) (64,29,32,64)

        # attention_weights = torch.softmax(t, dim=2) # (batch_size, max_cliques, n_elements) #之后返回注意力权重的时候，可以把权重映射到簇上面、映射到簇相应的原子上
        t, _ = torch.max(t, dim=2) #在第三个维度上取最大值，得到每个元素的最大值 # (batch_size, max_cliques, n_hidden_sets) (64, 29, 64) 32*64,64列中每列在32个数中选取最大值

        # ####### 求29个cliques的注意力分数
        # t_p = torch.normal(mean=0, std=1, size=(t.size(0), t.size(1), t.size(2))) # (64, 29, 64)
        t_p = t
        t_sum = torch.sum(t, dim=1) # (64, 64)
        t_sum = t_sum.unsqueeze(1) # (64, 1, 64)
        # t_sum_p = torch.normal(mean=0, std=1, size=(t.size(0), t.size(1), t.size(2))) # (64, 1, 64)
        # t_sum拓展为和t一样的维度，方便计算距离
        # t_sum_expand_p = t_sum_p.expand(-1, t.size(1), -1) # (64, 29, 64)，29行64维张量是一样的
        t_sum_expand_p = t_sum.expand(-1, t.size(1), -1) # (64, 29, 64)，29行64维张量是一样的
        # #计算每个元素和目标向量之间的距离
        distances = F.pairwise_distance(t_p, t_sum_expand_p, p=2) # (64, 29)
        # distances = F.cosine_similarity(t_p, t_sum_expand_p, dim=-1) # (64, 29)
        # distances = 1/(1+torch.exp(-distances)) # (64, 29)
        # 29维张量做min-max归一化
        attention_scores = (distances - distances.min(dim=1, keepdim=True)[0]) / (distances.max(dim=1, keepdim=True)[0] - distances.min(dim=1, keepdim=True)[0]) # (64, 29)
        # #######

        
        t_sum = torch.sum(t, dim=1) #在第二个维度上求和，得到每个分子的表示 # (batch_size, n_hidden_sets) (64, 64) 29个簇的64个特征按列相加求和，得到每个分子的64个特征
        t_sum  = self.bn(t_sum)
        t_f  = self.fc1(t_sum) # (batch_size, n_out_channels) (64, 300)
        out = self.relu(t_f)

        # t_sum = t_sum.unsqueeze(1) # (64, 1, 64)
        # #计算每个元素和目标向量之间的余弦相似度
        # attention_scores = F.cosine_similarity(t, t_sum, dim=-1) # (64, 29)

        # dot_product = torch.bmm(t, t_sum.transpose(1, 2)) # (batch_size, max_cliques, n_hidden_sets) (64, 29, 64) * (64, 64, 1) = (64, 29, 1)


        
        return out, attention_scores #处理集合数据，通过学习集合中元素的特征来表示整个集合
    
# self.Wc是一个参数化的权重矩阵，初始化时其形状是(d, n_hidden_sets * n_elements)。这意味着输入特征将被映射到一个较高维度的空间，其中包含多个隐藏集合特征。
# setrep可用于表示集合数据，通过学习集合中元素的特征来表示整个集合。可以把分子中的原子视为集合，这里把motif视为集合，motif相较于原子，具有更丰富的特征信息，因此用motif来表示分子，更具有优势。

# t是(64, 29, 64)
# t_sum = torch.sum(t, dim=1) # (64, 64)
# t_sum = t_sum.unsqueeze(2) # (64, 64, 1)
# ##########
# # 计算(64, 29, 64)和(64, 64)的之间的cosine相似度，得到(64, 29)
# tensor1_norm = torch.norm(t, dim=-1, p=2, keepdim=True) # (batch_size, max_cliques, n_hidden_sets) (64, 29, 1)
# tensor2_norm = torch.norm(t_sum, dim=1, p=2, keepdim=True) # (batch_size, n_hidden_sets) (64, 1, 1)
# # 计算点积
# dot_product = torch.bmm(t, t_sum) # (batch_size, max_cliques, n_hidden_sets) (64, 29, 64) * (64, 64, 1) = (64, 29, 1)
# # 计算余弦相似度
# attention_scores = dot_product / (tensor1_norm * tensor2_norm) # (batch_size, max_cliques, 1) (64, 29, 1)
# attention_scores = attention_scores.squeeze(-1) # (batch_size, max_cliques) (64, 29)
# ##########
