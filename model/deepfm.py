from .field_linear import FieldLinear 
from .field_embedding import FieldEmbedding 
from .pairwise_interaction import PairwiseInteraction 

import torch 
from torch import nn 


class DeepFM(nn.Module): 
    def __init__(self, field_dims, embedding_dims, out_features, hidden_units, dropout_rates): 
        super().__init__() 
        # *layers, out_layer = list(zip())


    def forward(self): 
        return 


class DNN(nn.Module): 
    def __init__(self, in_features, out_features, hidden_units, dropout_rates): 
        super().__init__() 


    def forward(self): 
        return 






