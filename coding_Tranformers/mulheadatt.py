import torch
import torch.nn as nn
import math


def scaled_dot_product(q, k, v, mask=None):                     # scaled dot product attention
    d_k = q.size()[-1]                                          # dimension of k
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)  # scaled dot product
    if mask is not None:                                            # mask              
        scaled += mask                                              # add mask
    attention = F.softmax(scaled, dim=-1)                           # attention
    values = torch.matmul(attention, v)                             # values
    return values, attention                                        # return values and attention                   

class MultiheadAttention(nn.Module):                                # multihead attention       

    def __init__(self, input_dim, d_model, num_heads):              # initialization
        super().__init__()                                          # super initialization                        
        self.input_dim = input_dim                                  # input dimension              
        self.d_model = d_model                                      # output dimension
        self.num_heads = num_heads                                  # number of heads             
        self.head_dim = d_model // num_heads                        # dimension of each head    
        self.qkv_layer = nn.Linear(input_dim , 3 * d_model)         # q, k, v layers
        self.linear_layer = nn.Linear(d_model, d_model)             # linear layer
    
    def forward(self, x, mask=None):                                # forward pass              
        batch_size, sequence_length, input_dim = x.size()           # batch size, sequence length, input dimension
        print(f"x.size(): {x.size()}")                                
        qkv = self.qkv_layer(x)                                     # qkv layer              
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)       # reshape
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)                               # permute                             
        print(f"qkv.size(): {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1)                              # chunk       
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ")
        values, attention = scaled_dot_product(q, k, v, mask)                                   # scaled dot product attention
        print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)    # reshape
        print(f"values.size(): {values.size()}")
        out = self.linear_layer(values)                                                         # linear layer                           
        print(f"out.size(): {out.size()}")
        return out                                                                               # return output tensor           
    
                        
                        
    # inputs
input_dim = 1024                                                # input dimension                       
d_model = 512                                                   # output dimension             
num_heads = 8                                                   # number of heads                     
batch_size = 30                                                 # batch size
sequence_length = 5                                             # sequence length
x = torch.randn( (batch_size, sequence_length, input_dim) )     # input tensor

   # outputs
model = MultiheadAttention(input_dim, d_model, num_heads)       # model
out = model.forward(x)                                          # forward pass    
                         