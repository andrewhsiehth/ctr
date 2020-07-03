from .layers import FeatureLinear 
from .layers import FeatureEmbedding  
from .layers import PairwiseInteraction 

import torch 
from torch import nn 

from typing import List

class DeepFM(nn.Module): 
    def __init__(self, num_fields: int, num_features: int, embedding_dim: int, out_features: int, hidden_units: List[int], dropout_rates: List[float]): 
        super().__init__() 
        self.feature_linear = FeatureLinear(num_features=num_features, out_features=out_features) 
        self.feature_embedding = FeatureEmbedding(num_features=num_features, embedding_dim=embedding_dim) 
        self.pairwise_interaction = PairwiseInteraction() 
        self.dnn = DNN(
            in_features=num_fields * embedding_dim, 
            out_features=out_features, 
            hidden_units=hidden_units, 
            dropout_rates=dropout_rates 
        )

    def forward(self, feature_idx, feature_value): 
        """
        :param feature_idx: (batch_size, num_fields)
        :param feature_value: (batch_size, num_fields)

        :return : (batch_size, out_features)
        """
        emb = self.feature_embedding(feature_idx, feature_value) 
        return self.feature_linear(feature_idx, feature_value) + self.pairwise_interaction(emb) + self.dnn(torch.flatten(emb, start_dim=1)) 


class DNN(nn.Module): 
    def __init__(self, in_features: int, out_features: int, hidden_units: List[int], dropout_rates: List[float]): 
        super().__init__() 
        *layers, out_layer = list(zip([in_features, *hidden_units], [*hidden_units, out_features])) 
        self.net = nn.Sequential(
            *(nn.Sequential(
                nn.Linear(in_features=i, out_features=o), 
                nn.BatchNorm1d(num_features=o), 
                nn.ReLU(), 
                nn.Dropout(p=p) 
            ) for (i, o), p in zip(layers, dropout_rates)), 
            nn.Linear(*out_layer)  
        )
    
    def forward(self, x): 
        """
        :param x: (batch_size, in_features) 

        :return : (batch_size, out_features)
        """
        return self.net(x)



