import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionMultiOutputNet(nn.Module):
    """
    Flexible Multi-Output ANN for Fashion Recommendation.
    """
    def __init__(self, input_dim, h1, h2, size_out, fit_out, shape_out, dropout_rate=0.2):
        super(FashionMultiOutputNet, self).__init__()
        
        # Shared Base: Now uses dynamic sizes (h1 and h2)
        self.shared_base = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # The Heads now correctly match the size of h2
        self.size_head = nn.Linear(h2, size_out)
        self.fit_head = nn.Linear(h2, fit_out)
        self.shape_head = nn.Linear(h2, shape_out)

    def forward(self, x):
        shared_features = self.shared_base(x)
        size_logits = self.size_head(shared_features)
        fit_logits = self.fit_head(shared_features)
        shape_logits = self.shape_head(shared_features)
        return size_logits, fit_logits, shape_logits
