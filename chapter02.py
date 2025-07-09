import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """Init method
        txt: string input text
        tokenizer: turns text into tokens
        max_length: int length of each input sentence
        stride: step size between chunks, how far to move sliding window"""

        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) # Tokenizes the entire text

        for i in range(0, len(token_ids) - max_length, stride): # Uses a sliding window
            input_chunk = token_ids[i:i + max_length] # Set of words of max length
            target_chunk = token_ids[i + 1: i + max_length + 1] # The next max length
            self.input_ids.append(torch.tensor(input_chunk)) # Stored as tensors
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids) # Returns the total number of rows from the dataset
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx] # Returns a single row from the dataset
    

def create_dataloader_v1(txt, batch_size=4, max_length=256,
    stride=128, shuffle=True, drop_last=True,
    num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2") # Tokenized

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) # Creates dataset

    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last, # True to avoid errors in training
    num_workers=num_workers
    )
    return dataloader