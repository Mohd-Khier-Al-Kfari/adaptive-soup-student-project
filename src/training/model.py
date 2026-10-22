"""Attached SelfHAR CNN and classifier. Author: Moh'd Khier Al Kfari."""
from __future__ import annotations
import torch
from torch import nn
class SelfHARCNN(nn.Module):
    """Attached three-block CNN with a dynamic activity head."""
    def __init__(self, input_shape: tuple[int, int], num_classes: int, dropout: float = .1):
        super().__init__(); channels = int(input_shape[1])
        self.features = nn.Sequential(nn.Conv1d(channels,64,5,padding=2,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(dropout),nn.Conv1d(64,128,5,padding=2,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(dropout),nn.Conv1d(128,96,5,padding=2,bias=False),nn.BatchNorm1d(96),nn.ReLU(),nn.Dropout(dropout))
        self.pool=nn.AdaptiveAvgPool1d(1); self.classifier=nn.Sequential(nn.Linear(96,1024),nn.ReLU(),nn.Linear(1024,int(num_classes)))
    def forward(self,x):
        if x.ndim==2: x=x.unsqueeze(0)
        return self.classifier(self.pool(self.features(x.transpose(1,2))).flatten(1))
