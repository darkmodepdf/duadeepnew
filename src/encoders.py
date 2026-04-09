import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class AntibodyEncoder(nn.Module):
    """
    Antibody Pathway: Utilizes IgBERT to generate robust embeddings for Heavy/Light chains.
    """
    def __init__(self, model_name="Exscientia/IgBERT", freeze=True):
        super(AntibodyEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
    def forward(self, sequences):
        # sequences: List of string representations of antibodies.
        tokens = self.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
        tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
        outputs = self.encoder(**tokens)
        
        # Typically use [CLS] token or mean pooling. Let's return mean pooled output.
        attention_mask = tokens['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return pooled


class AntigenEncoder(nn.Module):
    """
    Antigen Pathway: Utilizes ESM-Cambrian logic onto target antigens to glean 
    enriched biochemical features. Assumes Cambrian-style models accessible via HF.
    """
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", freeze=True):
        super(AntigenEncoder, self).__init__()
        # Adjust name if you have a local ESM-Cambrian model checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, sequences):
        tokens = self.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
        tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
        outputs = self.encoder(**tokens)
        
        attention_mask = tokens['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return pooled
