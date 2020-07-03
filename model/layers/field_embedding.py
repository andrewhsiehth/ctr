import torch 

from torch import nn 

from itertools import accumulate 

from typing import List 

class FieldEmbedding(nn.Module): 
    def __init__(self, field_dims: List[int], embedding_dim: int): 
        super().__init__() 
        self.weight = nn.Embedding(num_embeddings=sum(field_dims), embedding_dim=embedding_dim) 
        self.register_buffer('offset', torch.as_tensor([0, *accumulate(field_dims)][:-1], dtype=torch.long)) 

    def forward(self, x): 
        # x: (batch_size, num_fields)
        return self.weight(x + self.offset)
