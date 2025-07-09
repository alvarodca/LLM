import torch
import torch.nn as nn
from chapter03 import MultiHeadAttention


# Text generator
def generate_text_simple(model, idx,
    max_new_tokens, context_size):
    # Uses only the maximum amount of selected tokens
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # Takes the selected tokens as input
        with torch.no_grad(): # No gradient calculating (prediction)
            logits = model(idx_cond) # Output prediction 

        logits = logits[:, -1, :] # Logits for the last token (predicting the last wrod)
        probas = torch.softmax(logits, dim=-1) # Values to probabilities (softmax) not strictly necessary
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # Selects the highest value
        idx = torch.cat((idx, idx_next), dim=1) # Appends to the previous tokens
    return idx

# GELU
class GELU(nn.Module):
 def __init__(self):
    super().__init__()
    
 def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
    torch.sqrt(torch.tensor(2.0 / torch.pi)) *
    (x + 0.044715 * torch.pow(x, 3))
    ))
 
# FEED FORWARD
class FeedForward(nn.Module):
 def __init__(self, cfg):
    super().__init__()
    # Neural layers
    self.layers = nn.Sequential(
    nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # Increases embedding by 4
    GELU(),
    nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), # Decreases embedding by 4
    )
 def forward(self, x):
    return self.layers(x)

# TRANSFORMER BLOCK
class TransformerBlock(nn.Module):
 def __init__(self, cfg):
    super().__init__()
    """Init method
        d_in = input embedding size
        d_out = output embedding size
        context_length = max number of tokens in a sentence
        num_heads = number of heads to use 
        dropout = droput rate
        qkv_bias = whether to add bias"""
    self.att = MultiHeadAttention(
    d_in=cfg["emb_dim"],
    d_out=cfg["emb_dim"],
    context_length=cfg["context_length"],
    num_heads=cfg["n_heads"],
    dropout=cfg["drop_rate"],
    qkv_bias=cfg["qkv_bias"])

    # FeedForward structure
    self.ff = FeedForward(cfg)

    # Normalizes inputs before the different blocks <- PreLayerNorm
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])

    # Dropout layer
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
 
 def forward(self, x): # Enables shortcut connections

    # FIrst block of our previously shown picture
    shortcut = x
    x = self.norm1(x) # Normalize input
    x = self.att(x) # Self-attention
    x = self.drop_shortcut(x) # Dropout
    x = x + shortcut # Residual connection 

    # Second block
    shortcut = x
    x = self.norm2(x) # Normalize
    x = self.ff(x) # FeedForward    
    x = self.drop_shortcut(x) # Dropout
    x = x + shortcut # Residual 
    return x
 

# LAYERNORM
class LayerNorm(nn.Module):
 def __init__(self, emb_dim): # emb_dim is the last dimension, to work with these numbers
    super().__init__()
    self.eps = 1e-5 # Small value to avoid division by 0
    # Trainable parameters
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))
    
 def forward(self, x):
    # Applies normalization
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift

class GPTModel(nn.Module):
 def __init__(self, cfg):
    super().__init__()
    # Token and positional embedding layers, convert tokens into vectors and add valuable information
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])

    # Sequential Stack of Transformer Block equal to the number of layers
    self.trf_blocks = nn.Sequential(
    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

    # Layer Norm to standardize outputs
    self.final_norm = LayerNorm(cfg["emb_dim"])

    # Output into the vocabulary
    self.out_head = nn.Linear(
    cfg["emb_dim"], cfg["vocab_size"], bias=False
    )

 def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape
    # Computes embeddings
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(
    torch.arange(seq_len, device=in_idx.device)
    )
    # Adds embeddings
    x = tok_embeds + pos_embeds
    x = self.drop_emb(x) # Dropout
    x = self.trf_blocks(x) # Transformer Blocks
    x = self.final_norm(x) # Normalization
    logits = self.out_head(x) # Output
    return logits
 

