"""
Adapted from the Transfer Learning Library (TLlib):
Original author: Junguang Jiang
Modified for the ADA-SSR implementation.

The original source is distributed under the MIT License.
See the repository LICENSE_TLIB file for details.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class ConfidenceBasedSelfTrainingLoss(nn.Module):
    """
    Self training loss that adopts confidence threshold to select reliable pseudo labels from
    `Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks (ICML 2013)
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf>`_.

    Args:
        threshold (float): Confidence threshold.

    Inputs:
        - y: unnormalized classifier predictions.
        - y_target: unnormalized classifier predictions which will used for generating pseudo labels.

    Returns:
         A tuple, including
            - self_training_loss: self training loss with pseudo labels.
            - mask: binary mask that indicates which samples are retained (whose confidence is above the threshold).
            - pseudo_labels: generated pseudo labels.

    Shape:
        - y, y_target: :math:`(minibatch, C)` where C means the number of classes.
        - self_training_loss: scalar.
        - mask, pseudo_labels :math:`(minibatch, )`.

    """

    def __init__(self, threshold: float):
        super(ConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, y, y_target):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)
        mask = (confidence > self.threshold).float()
        self_training_loss = (F.cross_entropy(y, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels
    

"""
@author: Hajer Fradi
"""
# Another version with focal loss instead of cross entropy to better deal with class imabalnces    
def focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction='mean'):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # shape: [batch_size]
    pt = torch.exp(-ce_loss)  # pt = softmax probability of the true class
    focal = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    else:
        return focal

class ConfidenceBasedSelfTrainingLoss_focal(nn.Module):
    def __init__(self, threshold: float):
        super(ConfidenceBasedSelfTrainingLoss_focal, self).__init__()
        self.threshold = threshold

    def forward(self, y, y_target):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)
        mask = (confidence > self.threshold).float()
        self_training_loss = (focal_loss(y, pseudo_labels,alpha=0.5, gamma=2.5, reduction='none') * mask).mean()
        return self_training_loss, mask, pseudo_labels
