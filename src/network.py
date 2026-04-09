import torch
import torch.nn as nn
from encoders import AntibodyEncoder, AntigenEncoder

class DuaDeepImprovedInteractionHead(nn.Module):
    def __init__(self, ab_dim=768, ag_dim=1280, hidden_dims=[1024, 512, 128], num_classes=2):
        super(DuaDeepImprovedInteractionHead, self).__init__()
        
        # Cross Attention fusion layers could go here; 
        # using deep MLP concatenation mapping to adhere to original DuaDeep style
        fusion_dim = ab_dim + ag_dim
        
        layers = []
        in_dim = fusion_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = h_dim
            
        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, ab_embeds, ag_embeds):
        # Concatenate Antibody and Antigen embeddings
        fused = torch.cat([ab_embeds, ag_embeds], dim=-1)
        out = self.mlp(fused)
        logits = self.classifier(out)
        return logits


class DuaDeepImproved(nn.Module):
    """
    Combined Architecture mapping heterogeneous encoders through Interaction Core.
    """
    def __init__(self, ab_model="Exscientia/IgBERT", ag_model="facebook/esm2_t33_650M_UR50D", ab_freeze=True, ag_freeze=True):
        super(DuaDeepImproved, self).__init__()
        self.ab_encoder = AntibodyEncoder(model_name=ab_model, freeze=ab_freeze)
        self.ag_encoder = AntigenEncoder(model_name=ag_model, freeze=ag_freeze)
        
        # Dimensions are typical outputs for these model sizes. Adjust if needed.
        self.interaction_head = DuaDeepImprovedInteractionHead(ab_dim=768, ag_dim=1280)
        
    def forward(self, ab_seqs, ag_seqs):
        ab_emb = self.ab_encoder(ab_seqs)
        ag_emb = self.ag_encoder(ag_seqs)
        logits = self.interaction_head(ab_emb, ag_emb)
        return logits
