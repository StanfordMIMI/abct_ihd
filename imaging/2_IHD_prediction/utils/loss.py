"""
FocalLoss implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss as described here:
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=torch.tensor([0.1,0.9]), gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.alpha = alpha #class weights

        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.alpha = self.alpha.cuda(device=device)

        self.gamma = gamma #focusing parameter
        self.reduction = reduction # none | mean | sum

    def forward(self, scores, target):
        """
        Inputs
            -scores: torch tensors corresponding to unnormalized class scores of shape (N,C)
            -target: torch tensor of shape (N,)
        """
        log_prob = F.log_softmax(scores, dim=-1)
        prob = torch.exp(log_prob)

        return F.nll_loss(((1 - prob) ** self.gamma) * log_prob,
                            target,
                            weight=self.alpha,
                            reduction = self.reduction
                        )
