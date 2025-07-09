import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
 def __init__(self, d_in, d_out,
    context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()

    assert (d_out % num_heads == 0), \
    "d_out must be divisible by num_heads" # We need dimensions to align as we may lose dimensions in the process if not.

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads # Makes sure each attention head has equal dimensions

    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out) # Layer to combine outpute
    self.dropout = nn.Dropout(dropout)

    self.register_buffer(
    "mask",
    torch.triu(torch.ones(context_length, context_length),
    diagonal=1)
    )


 def forward(self, x):
    b, num_tokens, d_in = x.shape # b <- batch size, num_tokens <- sequence length, d_in <- input embedding

    # shape = [b,num_tokens, d_out] where d_out = num_heads * head_dim
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    # Split the values into Heads, shape = [b, num_tokens, num_heads, head_dim]
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(
    b, num_tokens, self.num_heads, self.head_dim
    )

    # As we are using nn.Linear we need to transpose our matrices
    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)

    # Computes dot product for each head
    attn_scores = queries @ keys.transpose(2, 3)

    # Masking
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # Mask fixed to token number
    attn_scores.masked_fill_(mask_bool, -torch.inf) # Uses mask to fill attention scores

    # Softmax + Dropout
    attn_weights = torch.softmax(
    attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    # Context vector
    context_vec = (attn_weights @ values).transpose(1, 2)

    # Combine all heads
    context_vec = context_vec.contiguous().view(
    b, num_tokens, self.d_out
    )
    # Mix information to a single simple layer
    context_vec = self.out_proj(context_vec)
    return context_vec