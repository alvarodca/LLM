# Loading weights into the gpt
import numpy as np
import torch
from chapter04 import generate_text_simple

# Assign function
def assign(left, right):
 if left.shape != right.shape:
    raise ValueError(f"Shape mismatch. Left: {left.shape}, "
 "Right: {right.shape}"
 )
 return torch.nn.Parameter(torch.tensor(right))

# Load weights function
def load_weights_into_gpt(gpt, params):
 gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
 gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

 for b in range(len(params["blocks"])):
    q_w, k_w, v_w = np.split(
    (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
    gpt.trf_blocks[b].att.W_query.weight = assign(
    gpt.trf_blocks[b].att.W_query.weight, q_w.T)
    gpt.trf_blocks[b].att.W_key.weight = assign(
    gpt.trf_blocks[b].att.W_key.weight, k_w.T)
    gpt.trf_blocks[b].att.W_value.weight = assign(
    gpt.trf_blocks[b].att.W_value.weight, v_w.T)
    q_b, k_b, v_b = np.split(
    (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
    gpt.trf_blocks[b].att.W_query.bias = assign(
    gpt.trf_blocks[b].att.W_query.bias, q_b)
    gpt.trf_blocks[b].att.W_key.bias = assign(
    gpt.trf_blocks[b].att.W_key.bias, k_b)
    gpt.trf_blocks[b].att.W_value.bias = assign(
    gpt.trf_blocks[b].att.W_value.bias, v_b)
    gpt.trf_blocks[b].att.out_proj.weight = assign(
    gpt.trf_blocks[b].att.out_proj.weight,
    params["blocks"][b]["attn"]["c_proj"]["w"].T)

    gpt.trf_blocks[b].att.out_proj.bias = assign(
    gpt.trf_blocks[b].att.out_proj.bias,
    params["blocks"][b]["attn"]["c_proj"]["b"])
    gpt.trf_blocks[b].ff.layers[0].weight = assign(
    gpt.trf_blocks[b].ff.layers[0].weight,
    params["blocks"][b]["mlp"]["c_fc"]["w"].T)
    gpt.trf_blocks[b].ff.layers[0].bias = assign(
    gpt.trf_blocks[b].ff.layers[0].bias,
    params["blocks"][b]["mlp"]["c_fc"]["b"])
    gpt.trf_blocks[b].ff.layers[2].weight = assign(
    gpt.trf_blocks[b].ff.layers[2].weight,
    params["blocks"][b]["mlp"]["c_proj"]["w"].T)
    gpt.trf_blocks[b].ff.layers[2].bias = assign(
    gpt.trf_blocks[b].ff.layers[2].bias,
    params["blocks"][b]["mlp"]["c_proj"]["b"])
    gpt.trf_blocks[b].norm1.scale = assign(
    gpt.trf_blocks[b].norm1.scale,
    params["blocks"][b]["ln_1"]["g"])
    gpt.trf_blocks[b].norm1.shift = assign(
    gpt.trf_blocks[b].norm1.shift,
    params["blocks"][b]["ln_1"]["b"])
    gpt.trf_blocks[b].norm2.scale = assign(
    gpt.trf_blocks[b].norm2.scale,
    params["blocks"][b]["ln_2"]["g"])
    gpt.trf_blocks[b].norm2.shift = assign(
    gpt.trf_blocks[b].norm2.shift,
    params["blocks"][b]["ln_2"]["b"])

 gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
 gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
 gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])



def text_to_token_ids(text, tokenizer):
 """Converts texts to token ids"""
 encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
 # Converts TokenIds into a Pytorch tensor
 encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
 return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
 "Changes ids to text"
 flat = token_ids.squeeze(0)
 return tokenizer.decode(flat.tolist()) # Converts back into text

def generate(model, idx, max_new_tokens, context_size,
 temperature=0.0, top_k=None, eos_id=None):
 # Same for loop as before
 for _ in range(max_new_tokens):
   idx_cond = idx[:, -context_size:]
   with torch.no_grad():
        logits = model(idx_cond)
        logits = logits[:, -1, :]
 
   # Top_k strategy
   if top_k is not None:
      top_logits, _ = torch.topk(logits, top_k)
      min_val = top_logits[:, -1]
      logits = torch.where(
      logits < min_val,
      torch.tensor(float('-inf')).to(logits.device),
      logits
      )
   # Temperature scaling
   if temperature > 0.0:
      logits = logits / temperature
      probs = torch.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
   else:
      # Greedy token selection if temperature is not used
      idx_next = torch.argmax(logits, dim=-1, keepdim=True)

   if idx_next == eos_id: # If end of sequence found, stops generating
      break
   idx = torch.cat((idx, idx_next), dim=1)
 return idx


# Training function
def train_model_simple(model, train_loader, val_loader,
 optimizer, device, num_epochs,
 eval_freq, eval_iter, start_context, tokenizer):
 """Function for training LLM model
 model: chosen NN model
 train_loader: DataLoader for training
 val_loader: DataLoader for testing
 optimizer: Optimization algorithm
 device: CPU or GPU
 num_epochs: amount of full passes through all data
 eval_freq: frequency of loss evaluation
 eval_iter: how many batches per evaluation
 start_context: Initial context prompt
 tokenizer: Model output into readable tokens"""

 # Initialize lists to track losses and seen tokens
 train_losses, val_losses, track_tokens_seen = [], [], []
 tokens_seen, global_step = 0, -1

 # Main training loop
 for epoch in range(num_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad() # Resets gradients

        # Calculates loss
        loss = calc_loss_batch(
        input_batch, target_batch, model, device
        )

        loss.backward() # Loss gradients
        optimizer.step() # Updates model weights
        tokens_seen += input_batch.numel() # TOtal tokens
        global_step += 1 # Taken update steps

        # Evaluate model periodically
        if global_step % eval_freq == 0:
            # Obtain loss
            train_loss, val_loss = evaluate_model(
            model, train_loader, val_loader, device, eval_iter)
            # Store loss and values
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            print(f"Ep {epoch+1} (Step {global_step:06d}): "
            f"Train loss {train_loss:.3f}, "
            f"Val loss {val_loss:.3f}"
            )

    # Sample text after each epoch
    generate_and_print_sample(
    model, tokenizer, device, start_context)


 return train_losses, val_losses, track_tokens_seen


# Implementing the required functions

# Prints training and validation losses to update them
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
 
 model.eval() # Dropout disappears
 
 with torch.no_grad(): # Evaluation is done, no need for gradients
    train_loss = calc_loss_loader(
    train_loader, model, device, num_batches=eval_iter
    )
    val_loss = calc_loss_loader(
    val_loader, model, device, num_batches=eval_iter
    )
 model.train() # Train the model
 return train_loss, val_loss

# Generates a text to validate its performance during training
def generate_and_print_sample(model, tokenizer, device, start_context):
 model.eval() # No training, just word generation
 context_size = model.pos_emb.weight.shape[0] # Maximum available context length
 encoded = text_to_token_ids(start_context, tokenizer).to(device) # Tokenizing text
 with torch.no_grad():
    token_ids = generate_text_simple(
    model=model, idx=encoded,
    max_new_tokens=50, context_size=context_size) # Generates 50 new tkens

 decoded_text = token_ids_to_text(token_ids, tokenizer) # Decoded tokens
 print(decoded_text.replace("\n", " "))
 model.train() # Back to 
 

# Cross entropy loss of a single batch
def calc_loss_batch(input_batch, target_batch, model, device):
 """ Input batch: batch of input data
 target batch: labels we wish to predict
 model: NN model
 device: CPU or GPU"""
 
 # Used more commonly for GPU
 input_batch = input_batch.to(device)
 target_batch = target_batch.to(device)

 # Output
 logits = model(input_batch)

 # Loss
 loss = torch.nn.functional.cross_entropy(
 logits.flatten(0, 1), target_batch.flatten()
 )
 return loss

# Loss over all batches
def calc_loss_loader(data_loader, model, device, num_batches=None):
 """ data_loader: DataLoader for providing batches of input
 model: evaluated model
 device: CPU or GPU
 num_batches: batches used to compute loss, if None, all batches are used"""
 
 # Initalize loss
 total_loss = 0.

 # Empty data
 if len(data_loader) == 0:
    return float("nan")
 
 # Define batches used
 elif num_batches is None:
    num_batches = len(data_loader)
 else:
    num_batches = min(num_batches, len(data_loader))
 # Loop over data, for each batch
 for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch( # COmputes loss
               input_batch, target_batch, model, device)
            total_loss += loss.item() # Adds the loss
        else:
            break

 # Returns average loss
 return total_loss / num_batches 


# Plot for visualizing loss and training
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
 fig, ax1 = plt.subplots(figsize=(5, 3))
 ax1.plot(epochs_seen, train_losses, label="Training loss")
 ax1.plot(
 epochs_seen, val_losses, linestyle="-.", label="Validation loss"
 )
 ax1.set_xlabel("Epochs")
 ax1.set_ylabel("Loss")
 ax1.legend(loc="upper right")
 ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
 ax2 = ax1.twiny()
 ax2.plot(tokens_seen, train_losses, alpha=0)
 ax2.set_xlabel("Tokens seen")
 fig.tight_layout()
 plt.show()
