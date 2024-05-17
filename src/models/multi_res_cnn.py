from typing import Optional
import math

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from gensim.models.word2vec import Word2Vec
import numpy as np

from src.models import BaseModel
from src.models.modules.attention import CAMLAttention
from src.text_encoders import BaseTextEncoder


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float,
    ):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=int(math.floor(kernel_size / 2)),
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int(math.floor(kernel_size / 2)),
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm1d(out_channels),
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


class MultiResCNN(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        num_classes_mimiciv_icd9: int,
        num_classes_mimiciv_icd10: int,
        num_filters: int,
        kernel_sizes: list[int],
        text_encoder: BaseTextEncoder,
        embed_dropout: float,
        pad_index: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_size = text_encoder.embedding_size
        self.embed_drop = nn.Dropout(p=embed_dropout)
        self.device = torch.device("cpu")
        self.num_classes_mimiciv_icd9 = num_classes_mimiciv_icd9
        self.num_classes_mimiciv_icd10 = num_classes_mimiciv_icd10

        self.loss = F.binary_cross_entropy_with_logits

        print("loading pretrained embeddings...")
        weights = torch.FloatTensor(text_encoder.weights)
        self.embed = nn.Embedding.from_pretrained(
            weights, padding_idx=pad_index, freeze=False
        )

        self.convs = nn.ModuleList()
        self.channels = nn.ModuleList()  # convolutional layers in parallel
        for kernel_size in kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.embed_size,
                        self.embed_size,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(math.floor(kernel_size / 2)),
                        bias=False,
                    ),
                    nn.Tanh(),
                    ResidualBlock(
                        self.embed_size, num_filters, kernel_size, 1, embed_dropout
                    ),
                )
            )

        self.attention = CAMLAttention(
            input_size=num_filters * len(self.convs), num_classes_icd9=self.num_classes_mimiciv_icd9, num_classes_icd10=self.num_classes_mimiciv_icd10
        )

    def forward(self, text):
        embedded = self.embed(text)
        embedded = self.embed_drop(embedded)
        embedded = embedded.transpose(1, 2)
        outputs = []
        for conv in self.convs:
            x = conv(embedded)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.attention(x)
        return x

    def get_loss(self, logits_icd9, logits_icd10, targets_icd9, targets_icd10, version):
        # Convert tuple version to PyTorch tensor
        version_tensor = torch.tensor([1 if v == 'icd9' else 2 for v in version])

        # Mask for separating ICD9 and ICD10 in the batch
        icd9_mask = (version_tensor == 1)
        icd10_mask = (version_tensor == 2)

        has_icd9_flag = icd9_mask.any()
        has_icd10_flag = icd10_mask.any()

        if not has_icd9_flag:
            icd10_loss = self.loss(logits_icd10[icd10_mask], targets_icd10[icd10_mask])
            return icd10_loss
        
        if not has_icd10_flag:
            icd9_loss = self.loss(logits_icd9[icd9_mask], targets_icd9[icd9_mask])
            return icd9_loss
        
        icd10_loss = self.loss(logits_icd10[icd10_mask], targets_icd10[icd10_mask], reduction='none').mean(dim=1)
        icd9_loss = self.loss(logits_icd9[icd9_mask], targets_icd9[icd9_mask], reduction='none').mean(dim=1)
        
        total_loss = torch.cat([icd9_loss, icd10_loss]).mean()

        return total_loss

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets_icd9, targets_icd10, _, version = batch.data, batch.targets_icd9, batch.targets_icd10, batch.num_tokens, batch.version
        logits_icd9, logits_icd10 = self(data)
        loss = self.get_loss(logits_icd9, logits_icd10, targets_icd9, targets_icd10, version)
        logits_icd9 = torch.sigmoid(logits_icd9)
        logits_icd10 = torch.sigmoid(logits_icd10)
        return {"logits_icd9": logits_icd9, "logits_icd10": logits_icd10, "loss": loss, "targets_icd9": targets_icd9, "targets_icd10": targets_icd10, "version": version}

    def validation_step(self, batch, version) -> dict[str, torch.Tensor]:
        data, targets_icd9, targets_icd10, _, version = batch.data, batch.targets_icd9, batch.targets_icd10, batch.num_tokens, batch.version
        logits_icd9, logits_icd10 = self(data)
        loss  = self.get_loss(logits_icd9, logits_icd10, targets_icd9, targets_icd10, version)
        logits_icd9 = torch.sigmoid(logits_icd9)
        logits_icd10 = torch.sigmoid(logits_icd10)
        return {"logits_icd9": logits_icd9, "logits_icd10": logits_icd10, "loss": loss, "targets_icd9": targets_icd9, "targets_icd10": targets_icd10, "version": version}
