import torch 

from torch import nn 


class PairwiseInteraction(nn.Module): 
    def __init__(self,): 
        super().__init__() 

    def forward(self, x): 
        # x: (batch_size, num_fields, embedding_dim) 

        square_of_sum = torch.square(torch.sum(x, dim=1)) 
        sum_of_square = torch.sum(torch.square(x), dim=1) 

        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True) 

