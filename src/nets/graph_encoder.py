import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, skip_input, *further_args):
        return skip_input + self.module(skip_input, *further_args)

class mySequential(nn.Sequential):
    """
    Source: https://github.com/pytorch/pytorch/issues/19808#issuecomment-487291323
    """
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        input_dim,
        embed_dim=None,
        val_dim=None,
        key_dim=None,
        shared_keyvalues=False,
        # one-to-one
        pred2succ=True,
        succ2pred=True,
        #one-to-many
        pred_group=False,
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        # Restricted Attentions Evaluation 
        self.pred2succ = pred2succ
        self.succ2pred = succ2pred
        self.pred_group = pred_group
        self.shared_keyvalues = shared_keyvalues

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_K = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_V = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # predecessor -> successor
        if self.pred2succ:
            self.W_Q_ps = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
            if not self.shared_keyvalues:
                self.W_K_ps = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
                self.W_V_ps = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # successor -> predecessor
        if self.succ2pred:
            self.W_Q_sp = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
            if not self.shared_keyvalues:
                self.W_K_sp = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
                self.W_V_sp = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # attention to precedence constraint group members
        if self.pred_group:
            self.W_Q_pg = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
            if not self.shared_keyvalues:
                self.W_K_pg = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
                self.W_V_pg = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, att_masks, group_masks, sparse_dist_masks, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)

        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_K).view(shp)
        V = torch.matmul(hflat, self.W_V).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        mask = sparse_dist_masks.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
        compatibility[~mask] = 0
        compatibilities = [compatibility]

        if self.pred2succ:
            Q_ps = torch.matmul(qflat, self.W_Q_ps).view(shp_q)
            if self.shared_keyvalues:
                K_ps = torch.matmul(hflat, self.W_K).view(shp)
                V_ps = torch.matmul(hflat, self.W_V).view(shp)
            else:
                K_ps = torch.matmul(hflat, self.W_K_ps).view(shp)
                V_ps = torch.matmul(hflat, self.W_V_ps).view(shp)

            compatibility_ps = self.norm_factor * torch.matmul(Q_ps, K_ps.transpose(2, 3))
            mask = att_masks.transpose(1,2).view(1, batch_size, n_query, graph_size).expand_as(compatibility_ps)
            compatibility_ps[~mask] = 0
            pos_ps = len(compatibilities)
            compatibilities.append(compatibility_ps)


        if self.succ2pred:
            Q_sp = torch.matmul(qflat, self.W_Q_sp).view(shp_q)
            if self.shared_keyvalues:
                K_sp = torch.matmul(hflat, self.W_K).view(shp)
                V_sp = torch.matmul(hflat, self.W_V).view(shp)
            else:
                K_sp = torch.matmul(hflat, self.W_K_sp).view(shp)
                V_sp = torch.matmul(hflat, self.W_V_sp).view(shp)

            compatibility_sp = self.norm_factor * torch.matmul(Q_sp, K_sp.transpose(2, 3))
            mask = att_masks.view(1, batch_size, n_query, graph_size).expand_as(compatibility_sp)
            compatibility_sp[~mask] = 0
            pos_sp = len(compatibilities)
            compatibilities.append(compatibility_sp)


        if self.pred_group:
            Q_pg = torch.matmul(qflat, self.W_Q_pg).view(shp_q)
            if self.shared_keyvalues:
                K_pg = torch.matmul(hflat, self.W_K).view(shp)
                V_pg = torch.matmul(hflat, self.W_V).view(shp)
            else:
                K_pg = torch.matmul(hflat, self.W_K_pg).view(shp)
                V_pg = torch.matmul(hflat, self.W_V_pg).view(shp)

            compatibility_pg = self.norm_factor * torch.matmul(Q_pg, K_pg.transpose(2, 3))
            mask = group_masks.view(1, batch_size, n_query, graph_size).expand_as(compatibility_pg)
            compatibility_pg[~mask] = 0
            pos_pg = len(compatibilities)
            compatibilities.append(compatibility_pg)


        compatibility = torch.cat(compatibilities, dim=-1)        
        
        sparse_compatibility = compatibility.to_sparse()
        attn = torch.sparse.softmax(sparse_compatibility, dim=3).to_dense()

        # attn = torch.softmax(compatibility, dim=-1)  # [n_heads, batch_size, n_query, graph_size+1+n_pick*2] (graph_size include depot)               

        heads = torch.matmul(attn[:, :, :, :graph_size], V)  # V: (self.n_heads, batch_size, graph_size, val_size)
        if self.pred2succ:
            heads += torch.matmul(attn[:, :, :, pos_ps*graph_size:(pos_ps+1)*graph_size], V_ps)
        if self.succ2pred:
            heads += torch.matmul(attn[:, :, :, pos_sp*graph_size:(pos_sp+1)*graph_size], V_sp)
        if self.pred_group:
            heads += torch.matmul(attn[:, :, :, pos_pg*graph_size:(pos_pg+1)*graph_size], V_pg)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for _, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden=512, normalization='batch',
        shared_keyvalues=None,
        pred2succ=None,
        succ2pred=None,
        pred_group=None
    ):
        super(MultiHeadAttentionLayer, self).__init__()
        
        self.mha = SkipConnection(MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    shared_keyvalues=shared_keyvalues,
                    pred2succ=pred2succ,
                    succ2pred=succ2pred,
                    pred_group=pred_group,
                )
            )
        self.norm0 = Normalization(embed_dim, normalization)
        self.ff = SkipConnection(nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            )
        self.norm1 = Normalization(embed_dim, normalization)

    def forward(self, x, att_masks, group_masks, sparse_dist_masks):
        out = self.mha(x, att_masks, group_masks, sparse_dist_masks)
        out = self.norm0(out)
        out = self.ff(out)
        out = self.norm1(out)
        return out, att_masks, group_masks, sparse_dist_masks # return masks for cascadenating


class GraphAttentionEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim, n_layers, node_dim=None,
        normalization='batch', feed_forward_hidden=512,
        shared_keyvalues=None,
        pred2succ=None,
        succ2pred=None,
        pred_group=None
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = mySequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization,
                shared_keyvalues=shared_keyvalues,
                pred2succ=pred2succ,
                succ2pred=succ2pred,
                pred_group=pred_group
            )
            for _ in range(n_layers)
        ))

    def forward(self, x, att_masks, group_masks, sparse_dist_masks):

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h, _, _, _ = self.layers(h, att_masks, group_masks, sparse_dist_masks)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
