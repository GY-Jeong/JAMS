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
from src.models.modules.attention import LabelAttention
from src.settings import PAD_TOKEN
from src.text_encoders import BaseTextEncoder


class LAAT(BaseModel):
    def __init__(
        self,
        num_classes_mimiciv_icd9: int,
        num_classes_mimiciv_icd10: int,
        text_encoder: BaseTextEncoder,
        embed_dropout: float = 0.5,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.5,
        projection_size: int = 128,
        pad_index: int = 0,
        **kwargs
    ):
        super(BaseModel, self).__init__()
        self.embed_size = text_encoder.embedding_size
        self.embed_drop = nn.Dropout(p=embed_dropout)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dropout = rnn_dropout
        self.projection_size = projection_size
        self.device = torch.device("cpu")
        self.num_classes_mimiciv_icd9 = num_classes_mimiciv_icd9
        self.num_classes_mimiciv_icd10 = num_classes_mimiciv_icd10

        self.loss = F.binary_cross_entropy_with_logits

        print("loading pretrained embeddings...")
        weights = torch.FloatTensor(text_encoder.weights)
        self.embed = nn.Embedding.from_pretrained(
            weights, padding_idx=pad_index, freeze=False
        )

        self.rnn = nn.LSTM(
            self.embed_size,
            self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            bidirectional=True,
            dropout=self.rnn_dropout,
        )

        self.attention = LabelAttention(
            input_size=self.rnn_hidden_size * 2,
            projection_size=self.projection_size,
            num_classes_icd9=num_classes_mimiciv_icd9,
            num_classes_icd10=num_classes_mimiciv_icd10
        )

    def encoder(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        return x

    def get_loss(self, logits_icd9, logits_icd10, targets_icd9, targets_icd10, version):
        # Convert tuple version to PyTorch tensor
        version_tensor = torch.tensor([1 if v == 'icd9' else 2 for v in version])

        # Mask for separating ICD9 and ICD10 in the batch
        icd9_mask = (version_tensor == 1)
        icd10_mask = (version_tensor == 2)

        has_icd9_flag = icd9_mask.any()
        has_icd10_flag = icd10_mask.any()

        # Compute loss for ICD9
        if not has_icd9_flag:
            # Compute loss for ICD10
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
        data, targets_icd9, targets_icd10, num_tokens, version = batch.data, batch.targets_icd9, batch.targets_icd10, batch.num_tokens, batch.version
        logits_icd9, logits_icd10 = self(data, num_tokens)
        loss = self.get_loss(logits_icd9, logits_icd10, targets_icd9, targets_icd10, version)
        logits_icd9 = torch.sigmoid(logits_icd9)
        logits_icd10 = torch.sigmoid(logits_icd10)
        return {"logits_icd9": logits_icd9, "logits_icd10": logits_icd10, "loss": loss, "targets_icd9": targets_icd9, "targets_icd10": targets_icd10, "version": version}

    def validation_step(self, batch, version) -> dict[str, torch.Tensor]:
        data, targets_icd9, targets_icd10, num_tokens, version = batch.data, batch.targets_icd9, batch.targets_icd10, batch.num_tokens, batch.version
        logits_icd9, logits_icd10 = self(data, num_tokens)
        loss  = self.get_loss(logits_icd9, logits_icd10, targets_icd9, targets_icd10, version)
        logits_icd9 = torch.sigmoid(logits_icd9)
        logits_icd10 = torch.sigmoid(logits_icd10)
        return {"logits_icd9": logits_icd9, "logits_icd10": logits_icd10, "loss": loss, "targets_icd9": targets_icd9, "targets_icd10": targets_icd10, "version": version}

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.rnn_num_layers * 2, batch_size, self.rnn_hidden_size).to(
                self.device
            ),
            torch.zeros(self.rnn_num_layers * 2, batch_size, self.rnn_hidden_size).to(
                self.device
            ),
        )

    def forward(self, x, num_tokens):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        embedding = self.encoder(x)
        self.rnn.flatten_parameters()
        embedding = pack_padded_sequence(
            embedding, num_tokens, batch_first=True, enforce_sorted=False
        )
        rnn_output, _ = self.rnn(embedding, hidden)
        rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
        output = self.attention(rnn_output)
        return output
