import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.init import xavier_normal_, uniform_, constant_
from .layers import TransformerBlock, PositionalEmbedding, CrossAttnTRMBlock


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(args,d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x ,mask, all_embeddings):
        for TRM in self.TRMs:
            x = TRM(x, mask=mask,all_embeddings = all_embeddings)
        return x


class Tokenizer(nn.Module):
    def __init__(self, rep_dim, vocab_size):
        super(Tokenizer, self).__init__()
        self.center = nn.Linear(rep_dim, vocab_size)

    def forward(self, x):
        bs, length, dim = x.shape
        probs = self.center(x.view(-1, dim))
        ret = F.gumbel_softmax(probs)
        indexes = ret.max(-1, keepdim=True)[1]
        return indexes.view(bs, length)


class Regressor(nn.Module):
    def __init__(self,args, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(args,d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


def get_batch_edge_index(org_edge_index, batch_num, node_num):
        # org_edge_index:(2, edge_num)
        edge_index = org_edge_index.clone().detach()
        edge_num = org_edge_index.shape[1]
        batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

        for i in range(batch_num):
            batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

        return batch_edge_index.long()
class DMASA(nn.Module):
    def __init__(self, args):
        super(DMASA, self).__init__()
        d_model = args.d_model

        # self.momentum = args.momentum
        self.linear_proba = True
        self.device = args.device
        self.data_shape = args.data_shape
        self.max_len = int(self.data_shape[0] / args.wave_length)
        # print(self.max_len)
        self.mask_len = int(args.mask_ratio * self.max_len)
        self.mask_ratio = args.mask_ratio
        self.position = PositionalEmbedding(self.max_len, d_model)

        self.mask_token = nn.Parameter(torch.randn(d_model, ))
        self.input_projection = nn.Conv1d(args.data_shape[1], d_model*args.data_shape[1], kernel_size=args.wave_length,
                                          stride=args.wave_length,groups=args.data_shape[1])
        self.encoder = Encoder(args)
        # self.momentum_encoder = Encoder(args)
        # self.tokenizer = Tokenizer(d_model, args.vocab_size)
        # self.reg = Regressor(args,d_model, args.attn_heads, 4 * d_model, 1, args.reg_layers)
        self.predict_head = nn.Linear(d_model, args.num_class)
        self.apply(self._init_weights)

        self.embedding = nn.Embedding(args.data_shape[1], d_model)
        self.topk = args.topk
        self.linear = nn.Linear(d_model*self.max_len,args.data_shape[0])
        self.linear1 = nn.Linear(args.data_shape[0], args.data_shape[0])
        self.relu = nn.ReLU(inplace = True)
        self.linear2 = nn.Linear(args.data_shape[0], args.data_shape[0])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data



    def pretrain_forward(self, x):
        ori = x
        batch_num, all_feature, node_num = x.shape
        device = x.device
        all_embeddings = self.embedding(torch.arange(node_num).to(device))
        weights_arr = all_embeddings.detach().clone()
        all_embeddings = all_embeddings.unsqueeze(0).unsqueeze(0)
        all_embeddings = all_embeddings.repeat(batch_num,self.max_len,1, 1)
        weights = weights_arr.view(node_num, -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        dim = weights.shape[-1]
        topk_num = self.topk
        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
        self.learned_graph = topk_indices_ji
        gmask = np.zeros((node_num,node_num))
        for i in range(node_num):
            for j in self.learned_graph[i]:
                gmask[i,j]=1
        self.gmask = torch.tensor(gmask).to(device)

        # mask = np.random.randint(0, 2, size=(self.data_shape[0], self.data_shape[1]))
        mask = np.random.choice([0, 1], size=(self.data_shape[0], self.data_shape[1]), p=[self.mask_ratio, 1 - self.mask_ratio])
        mask = torch.tensor(mask).to(device)
        x = x * mask
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        x = x.view(batch_num,self.max_len,-1, node_num)
        x = x.transpose(-1,-2)
        rep_visible = self.encoder(x,self.gmask,all_embeddings)
        rep_visible = rep_visible.transpose(2,1)
        rep_visible = rep_visible.reshape(batch_num,node_num,-1)
        rep = self.linear(rep_visible)
        rep = self.linear1(rep)
        rep = self.relu(rep)
        rep = self.linear2(rep)
        rep = rep.transpose(-1,-2)
        # rep_mask_prediction = self.reg(rep_visible, rep_mask_token)


        return [ori, rep]

    def draw(self,attention):
        import numpy as np
        import seaborn as sns
        # for i in range(128):
        # attention = attention[i]
        # uniform_data = torch.mean(attention, axis=(0, 1)).numpy()  # 自定义数据

        # mask = np.zeros_like(uniform_data)
        # mask[np.triu_indices_from(mask)] = True
        # with sns.axes_style("white"):mask = mask ,

        attention[attention<0.1] = 0
        ax = sns.heatmap(attention,cmap="YlGnBu", vmax=.3, square=True)
        plt.savefig('.//pic' + '//heat//heat1.eps', format='eps')
        plt.savefig('.//pic' + '//heat//heat1.svg', format='svg')
        plt.savefig('.//pic' + '//heat//heat1.png', format='png')
        plt.show()
        # ax = sns.heatmap(uniform_data)
        # heatmap = ax.pcolor(uniform_data, mask=mask)

    def forward(self, x):
        ori = x
        batch_num, all_feature, node_num = x.shape
        device = x.device
        all_embeddings = self.embedding(torch.arange(node_num).to(device))
        weights_arr = all_embeddings.detach().clone()
        all_embeddings = all_embeddings.unsqueeze(0).unsqueeze(0)
        all_embeddings = all_embeddings.repeat(batch_num, self.max_len, 1, 1)
        weights = weights_arr.view(node_num, -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        # self.draw(cos_ji_mat.detach().cpu())
        dim = weights.shape[-1]
        topk_num = self.topk
        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
        self.learned_graph = topk_indices_ji
        gmask = np.zeros((node_num, node_num))
        for i in range(node_num):
            for j in self.learned_graph[i]:
                gmask[i, j] = 1
        self.gmask = torch.tensor(gmask).to(device)

        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        x = x.view(batch_num, self.max_len, -1, node_num)
        x = x.transpose(-1, -2)
        rep_visible = self.encoder(x, self.gmask, all_embeddings)
        rep_visible = rep_visible.transpose(2, 1)
        rep_visible = rep_visible.reshape(batch_num, node_num, -1)
        rep = self.linear(rep_visible)
        rep = rep.transpose(-1, -2)
        # rep_mask_prediction = self.reg(rep_visible, rep_mask_token)

        return [ori, rep]

    def get_tokens(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        tokens = self.tokenizer(x)
        return tokens
