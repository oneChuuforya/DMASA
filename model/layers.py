import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from visualizer import get_local

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

def draw(attention):
    import numpy as np
    import seaborn as sns
    # for i in range(128):
        # attention = attention[i]
    uniform_data = torch.mean(attention,axis =(0,1) ).numpy() # 自定义数据

    # mask = np.zeros_like(uniform_data)
    # mask[np.triu_indices_from(mask)] = True
    # with sns.axes_style("white"):mask = mask ,
    ax = sns.heatmap(uniform_data,vmax=.3, square=True)
    plt.show()
    # ax = sns.heatmap(uniform_data)
    # heatmap = ax.pcolor(uniform_data, mask=mask)





class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, args, h, d_model ,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.node = args.data_shape[1]
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, query, key, value, mask=None,all_embeddings=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        #                      for l, x in zip(self.linear_layers, (query, key, value))]
        query, key, value = [l(x) for l, x in zip(self.linear_layers, (query, key, value))]#W*x
        # gquery, gkey, gvalue = query, key, value
        gquery, gkey, gvalue = torch.cat((all_embeddings,query),-1),torch.cat((all_embeddings,key),-1),torch.cat((value,all_embeddings),-1)
        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(gquery, gkey, value, mask=mask, dropout=self.dropout)
        x = self.relu(x)
        x = all_embeddings * x
        # 3) "Concat" using a view and apply a final linear.
        # x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.output_linear(x)

        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if type(x) == list:
            return self.norm(x[1] + self.dropout(self.a * sublayer(x)))
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x)))


class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    TRM layer
    """

    def __init__(self, args,d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(args,attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, x, mask,all_embeddings):
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x, _x, _x, mask=mask,all_embeddings = all_embeddings))
        x = self.skipconnect2(x, self.ffn)
        return x


class CrossAttnTRMBlock(nn.Module):
    def __init__(self,args, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(CrossAttnTRMBlock, self).__init__()
        self.attn = MultiHeadAttention(args,attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, rep_visible, rep_mask_token, mask=None):
        x = [rep_visible, rep_mask_token]
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x[1], _x[0], _x[0], mask=mask))
        x = self.skipconnect2(x, self.ffn)
        return x
