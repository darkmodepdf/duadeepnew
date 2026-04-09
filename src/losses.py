import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss designed to dynamically down-weight the loss contribution
    from massive over-represented clusters (e.g., HIV/COVID-19 antigen families) while
    focusing on harder, rarer classes.
    """
    def __init__(self, beta=0.9999, gamma=2.0, num_classes=2, samples_per_cls=None):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.samples_per_cls = samples_per_cls
        
        if samples_per_cls is not None:
            effective_num = 1.0 - torch.pow(self.beta, self.samples_per_cls)
            weights = (1.0 - self.beta) / effective_num
            weights = weights / torch.sum(weights) * self.num_classes
            self.register_buffer("weights", weights.float())
        else:
            self.weights = None

    def forward(self, labels, logits):
        """
        Compute CB-Focal loss.
        labels: Tensor of shape (batch_size,) with ground truth labels.
        logits: Tensor of shape (batch_size, num_classes) with raw model logits.
        """
        # Cross Entropy Loss computation
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        probs = torch.exp(-ce_loss)
        
        # Focal Loss
        focal_loss = (1 - probs) ** self.gamma * ce_loss
        
        # Apply class balancing weights if provided
        if self.weights is not None:
            batch_weights = self.weights[labels]
            focal_loss = batch_weights * focal_loss
            
        return focal_loss.mean()
