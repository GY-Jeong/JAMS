import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from torch.nn.init import xavier_uniform_

from src.models import BaseModel
from src.models.modules.attention import CAMLAttention
from src.text_encoders import BaseTextEncoder


class MullenbachBaseModel(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        num_classes_mimiciv_icd9: int,
        num_classes_mimiciv_icd10: int,
        text_encoder: BaseTextEncoder,
        embed_dropout: float = 0.5,
        pad_index: int = 0,
        **kwargs
    ):
        super(BaseModel, self).__init__()
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

    def encoder(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        return x.transpose(1, 2)

    def get_loss(self, logits, targets):
        return self.loss(logits, targets)


# TODO : modify
class CAML(MullenbachBaseModel):
    def __init__(
        self,
        vocab_size: int,
        num_classes_mimiciv_icd9: int,
        num_classes_mimiciv_icd10: int,
        text_encoder: Optional[Word2Vec] = None,
        embed_dropout: float = 0.2,
        pad_index: int = 0,
        num_filters: int = 500,
        kernel_size: int = 4,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_classes_mimiciv_icd9=num_classes_mimiciv_icd9,
            num_classes_mimiciv_icd10=num_classes_mimiciv_icd10,
            text_encoder=text_encoder,
            embed_dropout=embed_dropout,
            pad_index=pad_index,
        )

        self.conv = nn.Conv1d(
            self.embed_size,
            num_filters,
            kernel_size=kernel_size,
            padding=int(math.floor(kernel_size / 2)),
        )
        xavier_uniform_(self.conv.weight)

        self.attention = CAMLAttention(num_filters, num_classes_icd9=self.num_classes_mimiciv_icd9, num_classes_icd10=self.num_classes_mimiciv_icd10
        )

    def decoder(self, x):
        x = self.conv(x)
        return self.attention(x)

    def forward(self, x):
        representations = self.encoder(x)
        return self.decoder(representations)

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


