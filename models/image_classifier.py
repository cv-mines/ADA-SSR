"""
Adapted from the Transfer Learning Library (TLlib).
Original author: Baixu Chen
Modified for the ADA-SSR implementation.

The original source is distributed under the MIT License.
See the repository LICENSE_TLIB file for details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grl import WarmStartGradientReverseLayer
from .classifier import Classifier


class ImageClassifier(Classifier):
    r"""
    Classifier with non-linear pseudo head :math:`h_{\text{pseudo}}` and worst-case estimation head
    :math:`h_{\text{worst}}` from `Debiased Self-Training for Semi-Supervised Learning <https://arxiv.org/abs/2202.07136>`_.
    Both heads are directly connected to the feature extractor :math:`\psi`. We implement end-to-end adversarial
    training procedure between :math:`\psi` and :math:`h_{\text{worst}}` by introducing a gradient reverse layer.
    Note that both heads can be safely discarded during inference, and thus will introduce no inference cost.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer.
        width (int, optional): Hidden dimension of the non-linear pseudo head and worst-case estimation head.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - outputs: predictions of the main head :math:`h`
        - outputs_adv: predictions of the worst-case estimation head :math:`h_{\text{worst}}`
        - outputs_pseudo: predictions of the pseudo head :math:`h_{\text{pseudo}}`

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - outputs, outputs_adv, outputs_pseudo: (minibatch, `num_classes`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024, width=2048, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
        )
        bottleneck[0].weight.data.normal_(0, 0.005)
        bottleneck[0].bias.data.fill_(0.1)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        self.pseudo_head = nn.Sequential(
            nn.Linear(self.features_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, self.num_classes)
        )
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False)
        self.adv_head = nn.Sequential(
            nn.Linear(self.features_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, self.num_classes)
        )

    def forward(self, x: torch.Tensor):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        
        # NOTE:
        # The adversarial worst-case bias estimation introduced in DST [Chen et al., 2022]
        # is disabled in ADA-SSR, as adversarial learning is already handled by the
        # domain alignment component.
        # f_adv = self.grl_layer(f)
        # outputs_adv = self.adv_head(f_adv)
        outputs = self.head(f)
        outputs_pseudo = self.pseudo_head(f)
        if self.training:
            return outputs,f,outputs, outputs_pseudo
        else:
            return outputs

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.pseudo_head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.adv_head.parameters(), "lr": 1.0 * base_lr}
        ]

        return params

    def step(self):
        self.grl_layer.step()