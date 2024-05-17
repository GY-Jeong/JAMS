# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """
import torch
import torch.utils.checkpoint
from torch import nn

from transformers import RobertaModel, AutoConfig

from src.models.modules.attention import LabelAttention


class PLMICD(nn.Module):
    def __init__(self, num_classes_mimiciv_icd9: int, num_classes_mimiciv_icd10: int, model_path: str, **kwargs):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_path, finetuning_task=None
        )
        
        self.roberta = RobertaModel(
            self.config, add_pooling_layer=False
        ).from_pretrained(model_path, config=self.config)
        
        # TODO : lab
        self.attention = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            num_classes_icd9=num_classes_mimiciv_icd9,
            num_classes_icd10=num_classes_mimiciv_icd10
        )
        self.loss = torch.nn.functional.binary_cross_entropy_with_logits


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
        data, targets_icd9, targets_icd10, _, version, attention_mask = batch.data, batch.targets_icd9, batch.targets_icd10, batch.num_tokens, batch.version, batch.attention_mask
        logits_icd9, logits_icd10 = self(data, attention_mask)
        loss = self.get_loss(logits_icd9, logits_icd10, targets_icd9, targets_icd10, version)
        logits_icd9 = torch.sigmoid(logits_icd9)
        logits_icd10 = torch.sigmoid(logits_icd10)
        return {"logits_icd9": logits_icd9, "logits_icd10": logits_icd10, "loss": loss, "targets_icd9": targets_icd9, "targets_icd10": targets_icd10, "version": version}


    def validation_step(self, batch, version) -> dict[str, torch.Tensor]:
        data, targets_icd9, targets_icd10, _, version, attention_mask = batch.data, batch.targets_icd9, batch.targets_icd10, batch.num_tokens, batch.version, batch.attention_mask
        logits_icd9, logits_icd10 = self(data, attention_mask)
        loss = self.get_loss(logits_icd9, logits_icd10, targets_icd9, targets_icd10, version)
        logits_icd9 = torch.sigmoid(logits_icd9)
        logits_icd10 = torch.sigmoid(logits_icd10)
        return {"logits_icd9": logits_icd9, "logits_icd10": logits_icd10, "loss": loss, "targets_icd9": targets_icd9, "targets_icd10": targets_icd10, "version": version}
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        version=None,
        epoch=None
    ):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """

        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size)
            if attention_mask is not None
            else None,
            return_dict=False,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        logits = self.attention(hidden_output)
        return logits
