"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout
import torch.nn as nn
from einops import rearrange

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1
    
class TopKWindowAttention(nn.Module):
    def __init__(self,d_head,w=7,k=8,attention='linear'):
        super(TopKWindowAttention, self).__init__()
        self.w = w
        self.k = k
        self.d_head = d_head

        if attention == "linear":
            self.attention = LinearAttention()
        elif attention == "full":
            self.attention = FullAttention()
        else:
            raise NotImplementedError()

    def forward(self, q,k,v):
        b,d,h,w = q.shape
        qw = rearrange(q, 'b d (m w1) (n w2) -> b (m n) (w1 w2) d',w1=self.w,w2=self.w)
        kw = rearrange(k, 'b d (m w1) (n w2) -> b (m n) (w1 w2) d',w1=self.w,w2=self.w)
        vw = rearrange(v, 'b d (m w1) (n w2) -> b (m n) (w1 w2) d',w1=self.w,w2=self.w)
        qw_mean = torch.mean(qw,dim=2)
        kw_mean = torch.mean(kw,dim=2)
        vw_mean = torch.mean(vw,dim=2)
        
        window_similarity = torch.einsum('bmd,bnd->bmn',qw_mean,kw_mean)
        topk_values,topk_indices = torch.topk(window_similarity,dim=-1,k=self.k) # [b, m, k]
        
        fine_keys = []
        fine_values = []
        for i in range(b):
            fine_keys.append(kw[i][topk_indices[i]])
            fine_values.append(vw[i][topk_indices[i]])
        
        m,n = h // self.w, w // self.w
        fine_keys = torch.stack(fine_keys).reshape(b,m*n,-1,d) # [B, m*n, k*w1*w2, D]
        fine_values = torch.stack(fine_values).reshape(b,m*n,-1,d)
        
        keys = torch.cat([fine_keys,torch.tile(kw_mean.unsqueeze(1),(1,m*n,1,1))],2)
        values = torch.cat([fine_values,torch.tile(vw_mean.unsqueeze(1),(1,m*n,1,1))],2)
        
        queries = rearrange(qw,'b nw ws (h d) -> (b nw) ws h d',d=self.d_head)
        keys = rearrange(keys,'b nw ws (h d) -> (b nw) ws h d',d=self.d_head)
        values = rearrange(values,'b nw ws (h d) -> (b nw) ws h d',d=self.d_head)

        message = self.attention(queries, keys, values, q_mask=None, kv_mask=None)  # [N, L, (H, D)]
        message = rearrange(message, '(b m n) (w1 w2) h d -> b (h d) (m w1) (n w2)',m=m,n=n,w1=self.w)
        return message
    
class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()
