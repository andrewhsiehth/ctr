from .layers import FeatureLinear 
from .layers import FeatureEmbedding 
from .layers import PairwiseInteraction 

import torch 
from torch import nn 

from typing import List 

class FM(nn.Module): 
    def __init__(self, num_features: int, embedding_dim: int, out_features: int): 
        super().__init__() 
        self.feature_linear = FeatureLinear(num_features=num_features, out_features=out_features) 
        self.feature_embedding = FeatureEmbedding(num_features=num_features, embedding_dim=embedding_dim) 
        self.pairwise_interaction = PairwiseInteraction() 


    def forward(self, feature_idx, feature_value):  
        """
        :param feature_idx: (batch_size, num_fields) 
        :param feature_value: (batch_size, num_fields) 

        :return : (batch_size, out_features)
        """
        return self.feature_linear(feature_idx, feature_value) + self.pairwise_interaction(self.feature_embedding(feature_idx, feature_value)) 


