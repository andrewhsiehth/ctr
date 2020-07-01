from .field_linear import FieldLinear 
from .field_embedding import FieldEmbedding 

import torch 
from torch import nn 

from typing import List 

class AttentionalFM(nn.Module): 
    def __init__(self, field_dims: List[int], embedding_dim: int, attention_dim: int, out_features: int, dropout_rate: float): 
        super().__init__() 
        self.field_linear = FieldLinear(field_dims=field_dims, out_features=out_features) 
        self.field_embedding = FieldEmbedding(field_dims=field_dims, embedding_dim=embedding_dim) 
        self.attenional_interaction = AttentionalInteraction(
            embedding_dim=embedding_dim, 
            attention_dim=attention_dim, 
            out_features=out_features,
            dropout_rate=dropout_rate 
        )

    def forward(self, x): 
        """
        :param x: (batch_size, num_fields) 
        """
        return self.field_linear(x) + self.attenional_interaction(self.field_embedding(x)) 

class AttentionalInteraction(nn.Module): 
    def __init__(self, embedding_dim: int, attention_dim: int, out_features: int, dropout_rate: float): 
        super().__init__() 
        self.attention_score = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=attention_dim), 
            nn.ReLU(), 
            nn.Linear(in_features=attention_dim, out_features=1), 
            nn.Softmax(dim=1) 
        ) # (batch_size, num_fields * (num_fields - 1) / 2, 1)
        self.pairwise_product = PairwiseProduct() 
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features=embedding_dim, out_features=out_features) 

    def forward(self, x): 
        """
        :param x: (batch_size, num_fields, embedding_dim)
        """
        x = self.pairwise_product(x) # (batch_size, num_fields * (num_fields - 1) / 2, embedding_dim)
        score = self.attention_score(x) # (batch_size, num_fields * (num_fields - 1) / 2, 1)
        attentioned = torch.sum(score * x, dim=1) # (batch_size, embedding_dim) 
        return self.fc(self.dropout(attentioned)) # (batch_size, out_features)

class PairwiseProduct(nn.Module): 
    def __init__(self): 
        super().__init__() 
    
    def forward(self, x): 
        batch_size, num_fields, embedding_dim = x.size() 
        all_pairs_product = x.unsqueeze(dim=1) * x.unsqueeze(dim=2) # (batch_size, num_fields, num_fields, embedding_dim) 
        idx_row, idx_col = torch.unbind(torch.triu_indices(num_fields, num_fields, offset=1), dim=0) 
        return all_pairs_product[:, idx_row, idx_col] # (batch_size, num_fields * (num_fields - 1) / 2, embedding_dim) 



