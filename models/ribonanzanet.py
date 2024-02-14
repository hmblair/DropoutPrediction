# ribonanzanet.py

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
import os

class ScaledDotProductAttention(nn.Module):
    '''Scaled Dot-Product Attention'''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, attn_mask=None):
        attn = torch.matmul(q, k.transpose(2, 3))/ self.temperature
        if mask is not None:
            attn = attn+mask
        if attn_mask is not None:
            for i in range(len(attn_mask)):
                attn_mask[i,0]=attn_mask[i,0].fill_diagonal_(1)
            attn=attn.float().masked_fill(attn_mask == 0, float('-1e-9'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None,src_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  # For head axis broadcasting

        if src_mask is not None:
            src_mask=src_mask[:,:q.shape[2]].unsqueeze(-1).float()
            attn_mask=torch.matmul(src_mask,src_mask.permute(0,2,1))#.long()
            attn_mask=attn_mask.unsqueeze(1)
            q, attn = self.attention(q, k, v, mask=mask,attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
    


class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim : int,
        hidden_dim : Optional[int] = None,
        mix = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)


    def forward(self, x, src_mask = None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if src_mask is not None:
            src_mask=src_mask.unsqueeze(-1).float()
            src_mask = torch.matmul(src_mask,src_mask.permute(0,2,1))
            src_mask = rearrange(src_mask, 'b i j -> b i j ()')

        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)

        if src_mask is not None:
            left = left * src_mask
            right = right * src_mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)
    


class ConvTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, 
                 dim_feedforward, pairwise_dimension, dropout=0.1, k = 3,
                 ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)


        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.pairwise2heads=nn.Linear(pairwise_dimension,nhead,bias=False)
        self.pairwise_norm=nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()

        self.conv=nn.Conv1d(d_model,d_model,k,padding=k//2)

        self.outer_product_mean=OuterProductMean(
            in_dim=d_model,
            pairwise_dim=pairwise_dimension,
            )

        self.pair_transition=nn.Sequential(
            nn.LayerNorm(pairwise_dimension),
            nn.Linear(pairwise_dimension,pairwise_dimension*4),
            nn.ReLU(inplace=True),
            nn.Linear(pairwise_dimension*4,pairwise_dimension),
            )
        
        self.triangle_update_out=TriangleMultiplicativeModule(
            dim=pairwise_dimension,
            mix='outgoing',
            )
        self.triangle_update_in=TriangleMultiplicativeModule(
            dim=pairwise_dimension,
            mix='ingoing',
            )


    def forward(self, src , pairwise_features, src_mask=None, return_aw=False):
        
        if src_mask is not None:
            src = src * src_mask.float().unsqueeze(-1)

        src = src + self.conv(src.permute(0,2,1)).permute(0,2,1)
        src = self.norm3(src)

        pairwise_bias=self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0,3,1,2)
        src2,attention_weights = self.self_attn(src, src, src, mask=pairwise_bias, src_mask=src_mask)
        

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        pairwise_features = pairwise_features + self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.triangle_update_out(pairwise_features,src_mask)
        pairwise_features = pairwise_features + self.triangle_update_in(pairwise_features,src_mask)

        pairwise_features=pairwise_features+self.pair_transition(pairwise_features)
        if return_aw:
            return src,pairwise_features,attention_weights
        else:
            return src,pairwise_features



class OuterProductMean(nn.Module):
    def __init__(self, in_dim=256, dim_msa=32, pairwise_dim=64):
        super().__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa ** 2, pairwise_dim)

    def forward(self,seq_rep, pair_rep=None):
        seq_rep=self.proj_down1(seq_rep)
        outer_product = torch.einsum('bid,bjc -> bijcd', seq_rep, seq_rep)
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        outer_product = self.proj_down2(outer_product)

        if pair_rep is not None:
            outer_product=outer_product+pair_rep

        return outer_product 



class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.linear = nn.Linear(17, dim)


    def forward(self, src):
        L=src.shape[1]
        res_id = torch.arange(L).to(src.device).unsqueeze(0)
        device = res_id.device
        bin_values = torch.arange(-8, 9, device=device)
        d = res_id[:, :, None] - res_id[:, None, :]
        bdy = torch.tensor(8, device=device)
        d = torch.minimum(torch.maximum(-bdy, d), bdy)
        d_onehot = (d[..., None] == bin_values).float()
        assert d_onehot.sum(dim=-1).min() == 1
        p = self.linear(d_onehot)
        return p



class RibonanzaNet(nn.Module):
    def __init__(
            self, 
            ninp : int = 256,
            nhead : int = 8,
            nlayers : int = 9,
            ntoken : int = 5,
            nclass : int = 2,
            pairwise_dimension : int = 64,
            dropout : float = 0.05,
            ):
        super().__init__()
        self.transformer_encoder = []
        for i in range(nlayers):
            if i != nlayers-1:
                k=5
            else:
                k=1
            self.transformer_encoder.append(
                ConvTransformerEncoderLayer(
                    d_model = ninp, 
                    nhead = nhead,
                    dim_feedforward = ninp * 4, 
                    pairwise_dimension= pairwise_dimension,
                    dropout = dropout, 
                    k = k,
                    )
                )
        self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=4)
        self.decoder = nn.Linear(ninp, nclass)

        self.outer_product_mean=OuterProductMean(
            in_dim=ninp,
            pairwise_dim=pairwise_dimension,
            )
        self.pos_encoder=RelativePositionalEncoding(pairwise_dimension)
        self.embedding_dim = ninp
        if not os.path.exists('pretrained/checkpoints/ribonanzanet.pt'):
            raise FileNotFoundError(
                'Pretrained weights not found. Please download the weights from ... and place it in the pretrained/checkpoints/ directory.'
                )
        state_dict = torch.load(
            'pretrained/checkpoints/ribonanzanet.pt', 
            map_location=torch.device('cpu'),
            )
        self.load_state_dict(state_dict)


    def __getitem__(self, ix : int):
        return self.transformer_encoder[ix]
    

    def __len__(self):
        return len(self.transformer_encoder)


    def get_reactivity(self, src,src_mask=None,return_aw=False):
        src = self(src,src_mask=src_mask,return_aw=return_aw)
        output = self.decoder(src).squeeze(-1)
        return output
        
        
    def forward(self, src : torch.Tensor, src_mask=None, return_aw=False):
        B,L=src.shape
        src = src.long() - 4
        src = self.encoder(src).reshape(B,L,-1)
        pairwise_features=self.outer_product_mean(src)
        pairwise_features=pairwise_features+self.pos_encoder(src)
        for layer in self.transformer_encoder:
            src,pairwise_features=layer(src, pairwise_features, src_mask,return_aw=return_aw)
        return src