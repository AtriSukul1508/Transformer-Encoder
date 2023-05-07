import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product(q,k,v,mask=None):
  d_k = q.size()[-1]
  scaled = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k) # B x num_heads x M x M
  if mask is not None:
    scaled += mask
  attention = F.softmax(scaled,dim=-1) # B x num_heads x M x M
  values = torch.matmul(attention,v) # B x num_heads x M x head_dim
  return values

class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,num_heads):
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model/num_heads
    self.qkv_layer = nn.Linear(d_model,3*d_model)
    self.linear_layer = nn.Linear(d_model,d_model)
  def forward(self,input):
    batch_size, max_seq_len, d_model = input.size()
    qkv = self.qkv_layer(input) # B x M x 3*d_model
    qkv = qkv.reshape(batch_size,max_seq_len,self.num_heads,3*self.head_dim)
    qkv = qkv.permute(0,2,1,3)
    q,k,v = qkv.chunk(3,dim=-1) # B x num_heads x M x head_dim
    values = scaled_dot_product(q,k,v,mask) # B x num_heads x M x head_dim
    # values = values.permute(0,2,1,3)
    values = values.reshape(batch_size,max_seq_len,-1) # values.reshape(batch_size,max_seq_len,self.num_heads * self.head_dim)
    out = self.linear_layer(values)
    return out

class LayerNormalization(nn.Module):
  def __init__(self,parameter_shape,eps=1e-5):
    super().__init__()
    self.parameter_shape = parameter_shape #[d_model]
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(parameter_shape))
    self.beta = nn.Parameter(torch.ones(parameter_shape))
  def forward(self,input): # input -- B x m x d_model
    dims = [-(i+1) for i in range(len(self.parameter_shape))]
    mean = input.mean(dim=dims,keepdims=True) # B x M x 1
    var = ((input-mean)**2).mean(dim=dims,keepdims=True)
    std = (var+self.eps).sqrt()
    out = self.gamma * ((inputs-mean)/std) + self.beta
    return out
class PositionWiseFeedForward(nn.Module):
  def __init__(self,d_model,hidden_size,drop_prob=0.1):
    super().__init__()
    self.linear1 = nn.Linear(d_model,hidden_size)
    self.linear2 = nn.Linear(hidden_size,d_model)
    self.relu = nn.ReLU()
    self.dropout_layer = nn.Dropout(p=drop_prob)
  def forward(self,x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.dropout(x)
    out = self.linear2(x)
    return out

class SingleEncoderLayer(nn.Module):
  def __init__(self,d_model,hidden,num_heads,drop_prob):
    super().__init__()
    self.attention_layer = MultiHeadAttention(d_model,num_heads)
    self.layer_norm1 = LayerNormalization(parameter_shape=[d_model])
    self.feed_forward_layer = PositionWiseFeedForward(d_model,hidden,drop_prob)
    self.layer_norm2 = LayerNormalization(parameter_shape=[d_model])
    self.dropout_layer = nn.Dropout(p=drop_prob)
  def forward(self,x):
    residual = x
    x = self.attention_layer(x)
    x = self.dropout_layer(x)
    x = self.layer_norm1(x+residual)
    residual = x
    x = self.feed_forward_layer(x)
    x = self.dropout_layer(x)
    x = self.layer_norm2(x+residual)
    return x
class Encoder(nn.Module):
  def __init__(self,d_model,hidden,num_heads,drop_prob,num_layers):
    super().__init__()
    self.layers = nn.Sequential(*[SingleEncoderLayer(d_model,hidden,num_heads,drop_prob) for _ in range(num_layers)])
  def forward(self,x):
    return self.layers(x)