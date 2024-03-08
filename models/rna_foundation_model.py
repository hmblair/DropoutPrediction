# rna_foundation_model.py

import torch
import torch.nn as nn
import fm
from collections.abc import Sequence

class RNAFoundationModel(nn.Module, Sequence):
    """
    The RNA Foundation Model (RNA-FM) from the paper "Interpretable RNA 
    Foundation Model from Unannotated Data for Highly Accurate RNA Structure and
    Function Predictions", found at https://arxiv.org/abs/2204.00300.

    Attributes:
    ----------
        model (fm.pretrained.RNAFM): The pretrained RNA-FM model.
        embedding_dim (int): The dimension of the embeddings.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model, _ = fm.pretrained.rna_fm_t12() 
        self.model.eval()
        self.model.requires_grad_(False)
        self.embedding_dim = 640 


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Basic forward pass of the model that computes the RNA-FM embeddings.
        Feel free to override this method in your subclass to add additional 
        layers.

        Parameters:
        ----------
        x (torch.Tensor): 
            Input tensor.

        Returns:
        -------
        torch.Tensor: 
            Output tensor.
        """
        return self.compute_embeddings(x)


    def compute_embeddings(self, x : torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings for the given input tensor.

        Parameters:
        ----------
        x (torch.Tensor):
            Input tensor.

        Returns:
        -------
        torch.Tensor: 
            Embeddings computed for the input tensor.
        """
        # convert to long tensor for nn.Embedding
        x = x.long() 

        # apply the RNA-FM model
        x = self.model(x, repr_layers=[12]) 

        # return the embeddings
        return x["representations"][12] 
    

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute masked-language logits for the given tensor using the RNA-FM model.

        Parameters:
        ----------
        x (torch.Tensor):
             The input tensor.

        Returns:
        -------
        torch.Tensor: 
            The predicted logits.
        """
        # convert to long tensor for nn.Embedding
        x = x.long()  

        # apply the RNA-FM model
        x = self.model(x, repr_layers=[12])  

        # return the predicted logits
        return x['logits'] 
    

    def __getitem__(self, ix : int) -> nn.Module:
        """
        Get the layer at the given index.
        """
        return self.model.layers[ix]
    

    def __len__(self) -> int:
        """
        Return the number of layers in the model.
        """
        return len(self.model.layers)