# ribonanzanet.py

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
import os
import warnings

class ScaledDotProductAttention(nn.Module):
    """
    Implements a simple scaled dot-product attention module.

    Parameters:
    -----------
    temperature: float
        The temperature to scale the dot-product by.
    attn_dropout: float
        The dropout rate to apply to the attention weights.
    """
    def __init__(self, temperature : float, attn_dropout : float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
            self, 
            q : torch.Tensor, 
            k : torch.Tensor, 
            v : torch.Tensor, 
            mask : Optional[torch.Tensor] = None, 
            attn_mask : Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """
        Compute the scaled dot-product attention.

        Parameters:
        -----------
        q: torch.Tensor
            The query tensor.
        k: torch.Tensor
            The key tensor.
        v: torch.Tensor
            The value tensor.
        mask: torch.Tensor, optional
            The mask to apply to the attention weights. Defaults to None.
        attn_mask: torch.Tensor, optional
            The attention mask to apply to the attention weights. Defaults to None.

        Returns:
        --------
        torch.Tensor
            The output tensor.
        torch.Tensor
            The attention weights.
        """
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
    """
    Implements a multi-head attention module.

    Parameters:
    -----------
    d_model: int
        The number of input features.
    n_head: int
        The number of heads to use.
    d_k: int
        The dimensionality of the keys.
    d_v: int
        The dimensionality of the values.
    dropout: float
        The dropout rate to apply to the attention weights.
    """
    def __init__(
            self, 
            d_model : int,
            n_head : int, 
            d_k : int, 
            d_v : int, 
            dropout : float = 0.1,
            ) -> None:
        super().__init__()
        # store the number of heads, and the dimensionality of the keys and values
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # create the linear layers for the queries, keys and values
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        # create the attention module
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        # create the dropout layer 
        self.dropout = nn.Dropout(dropout)
        # create the layer normalisation layer
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(
            self, 
            q : torch.Tensor, 
            k : torch.Tensor, 
            v : torch.Tensor, 
            mask : Optional[torch.Tensor] = None, 
            src_mask : Optional[torch.Tensor] = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the multi-head attention between the query, key and value tensors.

        Parameters:
        -----------
        q: torch.Tensor
            The query tensor.
        k: torch.Tensor
            The key tensor.
        v: torch.Tensor
            The value tensor.
        mask: torch.Tensor, optional
            The mask to apply to the attention weights. Defaults to None.
        src_mask: torch.Tensor, optional
            The mask to apply to the source tensor. Defaults to None.

        Returns:
        --------
        torch.Tensor
            The output tensor.
        torch.Tensor
            The attention weights.
        """
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
    """
    Implements a triangle multiplicative module, as first described in the paper
    'Highly accurate protein structure prediction with AlphaFold' by Jumper et al.

    Parameters:
    -----------
    dim: int
        The number of input features.
    hidden_dim: int, optional
        The number of hidden features. Defaults to None.
    mix: str
        The mixing strategy to use. Must be either 'ingoing' or 'outgoing'.
    """
    def __init__(
        self,
        dim : int,
        hidden_dim : Optional[int] = None,
        mix : str = 'ingoing',
        ) -> None:
        super().__init__()
        # verify the mixing strategy
        if mix not in {'ingoing', 'outgoing'}:
            raise ValueError('mix must be either "ingoing" or "outgoing".')
        
        # store the mixing strategy
        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        # store the number of input features
        hidden_dim = hidden_dim if hidden_dim is not None else dim

        # create the layer normalisation layer
        self.norm = nn.LayerNorm(dim)

        # create the linear layers for the left and right projections
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        # create the linear layers for the left and right gates
        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be zero, with biases to be one
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        # create the layer normalisation and linear layers for the output
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)


    def forward(
            self, 
            x : torch.Tensor, 
            src_mask : Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """
        Compute the forward pass through the triangle multiplicative module.

        Parameters:
        -----------
        x: torch.Tensor
            The input tensor.
        src_mask: torch.Tensor, optional
            The mask to apply to the source tensor. Defaults to None.

        Returns:
        --------
        torch.Tensor
            The output tensor.
        """
        # verify the input shape
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'

        # mask the source tensor if necessary
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(-1).float()
            src_mask = torch.matmul(src_mask,src_mask.permute(0,2,1))
            src_mask = rearrange(src_mask, 'b i j -> b i j ()')

        # pass the input through the layer normalisation layer
        x = self.norm(x)

        # pass the input through the left and right projections
        left = self.left_proj(x)
        right = self.right_proj(x)

        # mask the left and right projections if necessary
        if src_mask is not None:
            left = left * src_mask
            right = right * src_mask

        # pass the input through the left and right gates
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        # apply the gating to the left and right projections
        left = left * left_gate
        right = right * right_gate

        # compute the outer product
        out = einsum(self.mix_einsum_eq, left, right)

        # pass the output through the layer normalisation and linear layers
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)
    


class ConvTransformerEncoderLayer(nn.Module):
    def __init__(
            self, 
            d_model : int, 
            nhead : int, 
            dim_feedforward : int, 
            pairwise_dimension : int, 
            dropout : float = 0.1, 
            k : int = 3,
            ) -> None:
        """
        A convolutional layer followed by a multi-head attention layer and a feedforward neural network.

        Parameters:
        -----------
        d_model: int
            The number of input features.
        nhead: int
            The number of heads to use in the multi-head attention module.
        dim_feedforward: int
            The number of hidden features to use in the feedforward neural network.
        pairwise_dimension: int
            The dimensionality of the pairwise features.
        dropout: float
            The dropout rate to apply to the attention weights.
        k: int
            The kernel size to use in the convolutional layer.
        """
        super().__init__()
        # initialize the multi-head attention module
        self.self_attn = MultiHeadAttention(
            d_model = d_model, 
            n_head = nhead, 
            d_k = d_model//nhead, 
            d_v = d_model//nhead, 
            dropout=dropout,
            )

        # initialize the feedforward neural networks for the encoder
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # initialize the layer normalisation and dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # initialize the pairwise attention module
        self.pairwise2heads=nn.Linear(pairwise_dimension,nhead,bias=False)
        self.pairwise_norm=nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()

        # initialize the convolutional layer
        self.conv=nn.Conv1d(
            in_channels = d_model, 
            out_channels = d_model, 
            kernel_size = k, 
            padding=k//2,
            )

        # initialize the outer product mean module
        self.outer_product_mean=OuterProductMean(
            in_dim=d_model,
            pairwise_dim=pairwise_dimension,
            )

        # initialize the pairwise transition module
        self.pair_transition=nn.Sequential(
            nn.LayerNorm(pairwise_dimension),
            nn.Linear(pairwise_dimension,pairwise_dimension*4),
            nn.ReLU(inplace=True),
            nn.Linear(pairwise_dimension*4,pairwise_dimension),
            )
        
        # initialize the triangle multiplicative modules
        self.triangle_update_out=TriangleMultiplicativeModule(
            dim=pairwise_dimension,
            mix='outgoing',
            )
        self.triangle_update_in=TriangleMultiplicativeModule(
            dim=pairwise_dimension,
            mix='ingoing',
            )


    def forward(
            self, 
            src : torch.Tensor, 
            pairwise_features : torch.Tensor, 
            src_mask : Optional[torch.Tensor] = None,
            return_aw : bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the forward pass through the transformer encoder layer.

        Parameters:
        -----------
        src: torch.Tensor
            The input tensor.
        pairwise_features: torch.Tensor
            The pairwise features tensor.
        src_mask: torch.Tensor, optional
            The mask to apply to the attention weights. Defaults to None.
        return_aw: bool, optional
            Whether to return the attention weights. Defaults to False.

        Returns:
        --------
        torch.Tensor
            The output tensor.
        torch.Tensor
            The updated pairwise features tensor.
        torch.Tensor, optional
            The attention weights tensor. Only returned if return_aw is True.
        """
        # mask the source tensor if necessary
        if src_mask is not None:
            src = src * src_mask.float().unsqueeze(-1)

        # pass the input through the convolutional layer
        src = src + self.conv(src.permute(0,2,1)).permute(0,2,1)

        # pass the input through the layer normalisation and dropout layers
        src = self.norm3(src)

        # pass the input through the pairwise attention module
        pairwise_bias=self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0,3,1,2)

        # pass the input through the multi-head attention module
        src2,attention_weights = self.self_attn(src, src, src, mask=pairwise_bias, src_mask=src_mask)
        
        # pass the input through the feedforward neural network
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # pass the input through the outer product mean module
        pairwise_features = pairwise_features + self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.triangle_update_out(pairwise_features,src_mask)
        pairwise_features = pairwise_features + self.triangle_update_in(pairwise_features,src_mask)

        # pass the input through the pairwise transition module
        pairwise_features=pairwise_features+self.pair_transition(pairwise_features)

        # return the output
        if return_aw:
            return src, pairwise_features, attention_weights
        else:
            return src, pairwise_features



class OuterProductMean(nn.Module):
    """
    An outer product mean module, as first described in the paper 'Highly 
    accurate protein structure prediction with AlphaFold' by Jumper et al.

    Parameters:
    -----------
    in_dim: int
        The number of input features.
    dim_msa: int
        The dimensionality of the MSA features.
    pairwise_dim: int
        The dimensionality of the pairwise features.
    """
    def __init__(
            self, 
            in_dim : int = 256, 
            dim_msa : int = 32, 
            pairwise_dim : int = 64,
            ) -> None:
        super().__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa ** 2, pairwise_dim)


    def forward(
            self,
            seq_rep : torch.Tensor, 
            pair_rep : Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """
        A forward pass through the outer product mean module.

        Parameters:
        -----------
        seq_rep: torch.Tensor
            The sequence representation tensor.
        pair_rep: torch.Tensor, optional
            The pairwise representation tensor. Defaults to None.

        Returns:
        --------
        torch.Tensor
            The outer product mean tensor.
        """
        # pass the input through the linear layers
        seq_rep=self.proj_down1(seq_rep)

        # compute the outer product
        outer_product = torch.einsum('bid,bjc -> bijcd', seq_rep, seq_rep)
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')

        # pass the outer product through the linear layers
        outer_product = self.proj_down2(outer_product)

        # add the pairwise representation tensor if necessary
        if pair_rep is not None:
            outer_product=outer_product+pair_rep

        return outer_product 



class RelativePositionalEncoding(nn.Module):
    """
    Implements a relative positional encoding.

    Parameters:
    -----------
    dim: int
        The dimensionality of the positional encoding.
    """
    def __init__(self, dim : int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(17, dim)


    def forward(self, src : torch.Tensor) -> torch.Tensor:
        """
        Add the relative positional encoding to the input tensor.

        Parameters:
        -----------
        src: torch.Tensor
            The input tensor.

        Returns:
        --------
        torch.Tensor
            The input tensor with the relative positional encoding added.
        """
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
    """
    The RibonanzaNet model.

    Parameters:
    -----------
    ninp: int
        The number of input features.
    nhead: int
        The number of heads to use in the multi-head attention module.
    nlayers: int
        The number of ConvTransformerEncoderLayer layers to use.
    ntoken: int
        The maximum token that can be used in the input tensor.
    nclass: int
        The number of classes to output.
    pairwise_dimension: int
        The dimensionality of the pairwise features.
    dropout: float
        The dropout rate to apply to the attention weights.
    """
    def __init__(
            self, 
            ninp : int = 256,
            nhead : int = 8,
            nlayers : int = 9,
            ntoken : int = 5,
            nclass : int = 2,
            pairwise_dimension : int = 64,
            dropout : float = 0.05,
            ) -> None:
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
        if not os.path.exists('models/checkpoints/ribonanzanet.pt'):
            warnings.warn(
                'The original RibonanzaNet weights were not found at models/checkpoints/ribonanzanet.pt. '
                'The model will be initialized with random weights.'
                'If you are loading another pre-trained model, feel free to ignore this warning.'
                )
        else:
            state_dict = torch.load(
                'models/checkpoints/ribonanzanet.pt', 
                map_location=torch.device('cpu'),
                )
            self.load_state_dict(state_dict)


    def __getitem__(self, ix : int) -> nn.Module:
        """
        Retrieve a layer from the model.
        """
        return self.transformer_encoder[ix]
    

    def __len__(self) -> int:
        """
        Get the number of layers in the model.
        """
        return len(self.transformer_encoder)


    def get_reactivity(
            self, 
            src : torch.Tensor,
            src_mask : Optional[torch.Tensor] = None,
            return_aw : bool = False,
            ) -> torch.Tensor:
        """
        A forward pass through the model to get the predicted reactivities.

        Parameters:
        -----------
        src: torch.Tensor
            The input tensor.
        src_mask: torch.Tensor, optional
            The mask to apply to the attention weights. Defaults to None. 
        return_aw: bool, optional
            Whether to return the attention weights. Defaults to False.

        Returns:
        --------
        torch.Tensor
            The predicted reactivities.
        """
        src = self(src,src_mask=src_mask,return_aw=return_aw)
        output = self.decoder(src).squeeze(-1)
        return output
        
        
    def forward(
            self, 
            src : torch.Tensor,
            src_mask : Optional[torch.Tensor] = None,
            return_aw : bool = False,
            ) -> torch.Tensor:       
        """
        A forward pass through the model, without the final linear layer.

        Parameters:
        -----------
        src: torch.Tensor
            The input tensor.
        src_mask: torch.Tensor, optional
            The mask to apply to the attention weights. Defaults to None. 
        return_aw: bool, optional
            Whether to return the attention weights. Defaults to False.

        Returns:
        --------
        torch.Tensor
            The embeddings of the input tensor. 
        """
        # get the shape of the input tensor
        B,L=src.shape

        # convert the input tensor to long
        src = src.long()

        # pass the input through the embedding layer
        src = self.encoder(src).reshape(B,L,-1)
        pairwise_features=self.outer_product_mean(src)
        pairwise_features=pairwise_features+self.pos_encoder(src)
        for layer in self.transformer_encoder:
            src,pairwise_features=layer(src, pairwise_features, src_mask,return_aw=return_aw)
        return src