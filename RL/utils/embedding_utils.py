import os
from typing import List

import ray
import torch
from transformers import AutoModel, AutoTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@ray.remote(num_cpus=1, num_gpus=1)
class EmbeddingWorker:
    def __init__(self, model_name: str, tokenizer_path: str):
        """
        Initializes the ModelWorker by loading the tokenizer and model.
        The model is set to half-precision and moved to the CUDA device assigned by Ray.

        Args:
            model_name (str): The name of the Hugging Face model to load.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set the pad token to the end-of-sequence token

        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.eval()
        self.device = torch.device('cuda')
        self.model.to(self.device)

    def compute_last_hidden_state(self, texts: tuple[str, ...]) -> List[List[float]]:
        texts = list(texts)
        # Tokenize input texts with padding and truncation
        inputs = self.tokenizer(
            texts,
            padding='max_length',      # Pad sequences to the max_length
            truncation=True,           # Truncate sequences longer than max_length
            max_length=512,            # Define the maximum sequence length
            return_tensors="pt"        # Return PyTorch tensors
        )
        
        # Move input tensors to the CUDA device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Forward pass through the model without computing gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract the last hidden state tensor
        # Shape: [batch_size, sequence_length, hidden_size]
        last_hidden_state = outputs.last_hidden_state  # Example shape: [batch_size, 512, 768]
        
        # Extract the attention mask to identify non-padded tokens
        # Shape: [batch_size, sequence_length]
        attention_mask = inputs['attention_mask']  # Example shape: [batch_size, 512]

        # Expand the attention mask dimensions for broadcasting
        # Shape after unsqueeze: [batch_size, sequence_length, 1]
        mask = attention_mask.unsqueeze(-1)  # Shape: [batch_size, 512, 1]

        # Apply the mask to the last hidden state to zero out padded token embeddings
        # Shape: [batch_size, sequence_length, hidden_size]
        masked_hidden = last_hidden_state * mask

        # Sum the masked hidden states across the sequence length
        # Shape: [batch_size, hidden_size]
        sum_hidden = masked_hidden.sum(dim=1)  # Shape: [batch_size, 768]

        # Compute the number of non-padded tokens for each input in the batch
        # Shape: [batch_size, 1]
        num_tokens = mask.sum(dim=1)  # Shape: [batch_size, 1]

        # To avoid division by zero, replace zero counts with one
        num_tokens = num_tokens.masked_fill(num_tokens == 0, 1)

        # Compute the mean by dividing the summed hidden states by the number of tokens
        # Shape: [batch_size, hidden_size]
        mean_hidden = sum_hidden / num_tokens  # Shape: [batch_size, 768]

        # Move the tensor to CPU and convert to NumPy array for serialization
        return mean_hidden.cpu().numpy().tolist()
    # Helper function to submit a batch and return its start index with embeddings
    def submit_batch(self, batch):
        ids, texts = zip(*batch)
        embeddings = self.compute_last_hidden_state(texts)
        return zip(ids, embeddings)
