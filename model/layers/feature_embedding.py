import torch 

from torch import nn 

class FeatureEmbedding(nn.Module): 
    def __init__(self, num_features: int, embedding_dim: int): 
        super().__init__() 
        self.weight = nn.Embedding(num_embeddings=num_features, embedding_dim=embedding_dim) 

    def forward(self, feature_idx, feature_value): 
        """
        :param feature_idx: (batch_size, num_fields) 
        :param feature_value: (batch_size, num_fields) 

        :return : (batch_size, num_fields, embedding_dim) 
        """
        return self.weight(feature_idx) * feature_value.unsqueeze(dim=-1) 
