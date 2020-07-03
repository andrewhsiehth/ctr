import torch 

from torch import nn 

class FeatureLinear(nn.Module): 
    def __init__(self, num_features: int, out_features: int): 
        super().__init__() 
        self.weight = nn.Embedding(num_embeddings=num_features, embedding_dim=out_features) 
        self.bias = nn.Parameter(torch.zeros((out_features,))) 
    
    def forward(self, feature_idx, feature_value): 
        """
        :param feature_idx: (batch_size, num_fields) 
        :param feature_value: (batch_size, num_fields) 

        :return : (batch_size, out_features) 
        """
        return torch.sum(self.weight(feature_idx) * feature_value.unsqueeze(dim=-1), dim=1) + self.bias  

