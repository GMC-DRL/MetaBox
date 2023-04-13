

import torch
import torch.nn.functional as F
from torch import nn
import math


# implements skip-connection module
class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


# implements MLP module
class MLP(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 mid_dim1,
                 mid_dim2,
                 output_dim,
                 p=0
                 ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, mid_dim1)
        self.fc2 = torch.nn.Linear(mid_dim1, mid_dim2)
        self.fc3 = torch.nn.Linear(mid_dim2, output_dim)

        self.dropout = torch.nn.Dropout(p=p)
        # func_choice={
        #     'ReLU':nn.ReLU,
        #     'Tanh':nn.Tanh,
        #     'Sigmoid':nn.Sigmoid,
        #     'LeakyReLU':nn.LeakyReLU,
        # }
        self.activation1 = nn.LeakyReLU(0.3)
        # self.init_parameters()

    # def init_parameters(self):
    #     for param in self.parameters():
    #         stdv = 1. / math.sqrt(param.size(-1))
    #         param.data.uniform_(-stdv, stdv)

    def forward(self, in_):
        result = self.activation1(self.fc1(in_))
        result = self.activation1(self.fc2(result))
        # result = self.dropout(result)
        result = self.fc3(result).squeeze(-1)
        return result


# implements Normalization module
class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()
        # 选择一种norm的方法
        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalization = normalization

        if not self.normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim, affine=True)

    # def init_parameters(self):

    #     for name, param in self.named_parameters():
    #         stdv = 1. / math.sqrt(param.size(-1))
    #         param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.normalization == 'layer':
            # out= (input-input.min(dim=-2,keepdim=True)[0]) / (torch.max(input,dim=-2,keepdim=True)[0]-torch.min(input,dim=-2,keepdim=True)[0]+1e-5)
            out = (input - input.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(input.var((1, 2)).view(-1, 1, 1) + 1e-05)
            return out

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


# implements the encoder for Critic net
class MultiHeadAttentionLayerforCritic(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionLayerforCritic, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )

    # implements the decoder for Critic net


class ValueDecoder(nn.Module):
    def __init__(
            self,
            input_dim,
            out_dim,
            ac_func
    ):
        super(ValueDecoder, self).__init__()
        # self.mlp=MLP(input_dim,output_dim=out_dim)
        self.ac_func = ac_func
        self.mlp = MLP(input_dim, 32, 16, 1, 0, ac_func)

    def forward(self, h_em):
        # mean 掉ps维度
        h_em = torch.mean(h_em, dim=1)
        value = self.mlp(h_em)
        if self.ac_func == 'Tanh':
            value = (value + 1.) / 2.
        return value


# implements the orginal Multi-head Self-Attention module
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        # 一般来说input_dim 和 embed_dim 是一样的？
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        # self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_query = nn.Parameter(torch.rand((n_heads, input_dim, key_dim)))
        self.W_key = nn.Parameter(torch.rand((n_heads, input_dim, key_dim)))
        self.W_val = nn.Parameter(torch.rand((n_heads, input_dim, val_dim)))

        if embed_dim is not None:
            # self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            self.W_out = nn.Parameter(torch.rand((n_heads, key_dim, embed_dim)))

        # self.init_parameters()

    # def init_parameters(self):

    #     for param in self.parameters():
    #         stdv = 1. / math.sqrt(param.size(-1))
    #         param.data.uniform_(-stdv, stdv)

    def forward(self, h, q=None):
        if q == None:
            q = h  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)

        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


# implements the multi-head compatibility layer
# 只算相似性
class MultiHeadCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadCompat, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(1 * key_dim)

        # self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_query = nn.Parameter(torch.rand((n_heads, input_dim, key_dim)))
        self.W_key = nn.Parameter(torch.rand((n_heads, input_dim, key_dim)))

        # self.init_parameters()

    # def init_parameters(self):

    #     for param in self.parameters():
    #         stdv = 1. / math.sqrt(param.size(-1))
    #         param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = torch.matmul(Q, K.transpose(2, 3))

        return self.norm_factor * compatibility


# implements the encoder
class MultiHeadEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadEncoder, self).__init__()

        self.MHA_sublayer = MultiHeadAttentionsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFandNormsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input, input_=None):
        out = self.MHA_sublayer(input, input_)
        return self.FFandNorm_sublayer(out)


# implements the encoder (DAC-Att sublayer)
# 还加入了残差连接
class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()

        self.MHA = MultiHeadAttention(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input, input_=None):
        # Attention and Residual connection
        out = self.MHA(input, input_)

        # Normalization
        return self.Norm(out + input)


# implements the encoder (FFN sublayer)
class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayer, self).__init__()

        self.FF = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_hidden, embed_dim)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input):
        # FF and Residual connection
        out = self.FF(input)

        # Normalization
        return self.Norm(out + input)


class EmbeddingNet(nn.Module):

    def __init__(
            self,
            node_dim,
            embedding_dim,
    ):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias=False)

    # def init_parameters(self):

    #     for param in self.parameters():
    #         stdv = 1. / math.sqrt(param.size(-1))
    #         param.data.uniform_(-stdv, stdv)

    def forward(self, x):
        h_em = self.embedder(x)
        return h_em
