import numpy as np
import math

# parameters
L, d_k, d_v = 4, 8, 8
q = np.random.randn(L, d_k)
k = np.random.randn(L, d_k)
v = np.random.randn(L, d_v)

# mask
mask = np.tril(np.ones( (L, L) ))

# softmax
def softmax(x):
      return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T   # axis=-1 means last axis

def scaled_dot_product_attention(q, k, v, mask=None):       # pylint: disable=protected-access
  d_k = q.shape[-1]                                         # last dimension of q
  scaled = np.matmul(q, k.T) / math.sqrt(d_k)               # (L, L)        
  if mask is not None:                                      # mask
    scaled = scaled + mask                                  # scale                
  attention = softmax(scaled)                               # (L, L)
  out = np.matmul(attention, v)                             # (L, d_v)                      
  return out, attention                                     # (L, d_v), (L, L)
     

values, attention = scaled_dot_product_attention(q, k, v, mask=mask)    # (L, d_v), (L, L)
print("Q\n", q) 
print("K\n", k)
print("V\n", v)
print("New V\n", values)
print("Attention\n", attention)