import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_

from src.models.modules.graph_attention import GraphAttentionLayer

    
class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes_icd9: int, num_classes_icd10: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)

        # TODO : label vector 수정
        # self.second_linear_icd9 = nn.Linear(projection_size, num_classes_icd9, bias=False)
        # self.second_linear_icd10 = nn.Linear(projection_size, num_classes_icd10, bias=False)

        self.label_vector = GraphAttentionLayer(in_features=projection_size,
                                                out_features=projection_size,
                                                num_classes_icd9=num_classes_icd9,
                                                num_classes_icd10=num_classes_icd10,
                                                dropout=0.2,
                                                alpha=0.2)

        self.third_linear_icd9 = nn.Linear(input_size, num_classes_icd9)
        self.third_linear_icd10 = nn.Linear(input_size, num_classes_icd10)
        self._init_weights(mean=0.0, std=0.03)

        # self.layer_norm_icd9 = nn.LayerNorm([num_classes_icd9])
        # self.layer_norm_icd10 = nn.LayerNorm([num_classes_icd10])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        # Z = tanh(WH)
        weights = torch.tanh(self.first_linear(x))

        label_vector_icd9 = self.label_vector('mimiciv_icd9').unsqueeze(0)
        label_vector_icd10 = self.label_vector('mimiciv_icd10').unsqueeze(0)

        att_weights_icd9 = torch.matmul(weights, label_vector_icd9.transpose(-1, -2))
        att_weights_icd10 = torch.matmul(weights, label_vector_icd10.transpose(-1, -2))
        
        # att_weights_icd9 = self.layer_norm_icd9(att_weights_icd9)
        # att_weights_icd10 = self.layer_norm_icd10(att_weights_icd10)

        att_weights_icd9 = torch.nn.functional.softmax(att_weights_icd9, dim=1).transpose(1, 2)
        att_weights_icd10 = torch.nn.functional.softmax(att_weights_icd10, dim=1).transpose(1, 2)

        weighted_output_icd9 = att_weights_icd9 @ x
        weighted_output_icd10 = att_weights_icd10 @ x

        logits_icd9 = self.third_linear_icd9.weight.mul(weighted_output_icd9).sum(dim=2).add(self.third_linear_icd9.bias)
        logits_icd10 = self.third_linear_icd10.weight.mul(weighted_output_icd10).sum(dim=2).add(self.third_linear_icd10.bias)

        return logits_icd9, logits_icd10


    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear_icd9.weight, mean, std)
        torch.nn.init.normal_(self.third_linear_icd10.weight, mean, std)


class CAMLAttention(nn.Module):
    def __init__(self, input_size: int, num_classes_icd9: int, num_classes_icd10: int):
        super().__init__()
        
        self.label_vector = GraphAttentionLayer(in_features=input_size,
                                                out_features=input_size,
                                                num_classes_icd9=num_classes_icd9,
                                                num_classes_icd10=num_classes_icd10,
                                                dropout=0.2,
                                                alpha=0.2)
        
        self.second_linear_icd9 = nn.Linear(input_size, num_classes_icd9)
        self.second_linear_icd10 = nn.Linear(input_size, num_classes_icd10)

        xavier_uniform_(self.second_linear_icd9.weight)
        xavier_uniform_(self.second_linear_icd10.weight)
        
        # self.layer_norm_icd9 = nn.LayerNorm([num_classes_icd9])
        # self.layer_norm_icd10 = nn.LayerNorm([num_classes_icd10])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CAML attention mechanism

        Args:
            x (torch.Tensor): [batch_size, input_size, seq_len]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        x = torch.tanh(x)

        label_vector_icd9 = self.label_vector('mimiciv_icd9')
        # [icd10 개수, input_size]
        label_vector_icd10 = self.label_vector('mimiciv_icd10')

        # [batch_size, icd10 개수, input_size]
        att_weights_icd9 = label_vector_icd9.matmul(x)
        # [batch_size, input_size, icd10 개수]
        att_weights_icd9 = att_weights_icd9.transpose(1, 2)
        # att_weights_icd9 = self.layer_norm_icd9(att_weights_icd9)
        att_weights_icd9 = att_weights_icd9.transpose(1, 2)
        att_weights_icd9 = torch.softmax(att_weights_icd9, dim=2)
        weighted_output_icd9 = att_weights_icd9 @ x.transpose(1, 2)

        att_weights_icd10 = label_vector_icd10.matmul(x)
        att_weights_icd10 = att_weights_icd10.transpose(1, 2)
        # att_weights_icd10 = self.layer_norm_icd10(att_weights_icd10)
        att_weights_icd10 = att_weights_icd10.transpose(1, 2)
        att_weights_icd10 = torch.softmax(att_weights_icd10, dim=2)
        weighted_output_icd10 = att_weights_icd10 @ x.transpose(1, 2)
        logits_icd9 = self.second_linear_icd9.weight.mul(weighted_output_icd9).sum(dim=2).add(self.second_linear_icd9.bias)
        logits_icd10 = self.second_linear_icd10.weight.mul(weighted_output_icd10).sum(dim=2).add(self.second_linear_icd10.bias)
        
        return logits_icd9, logits_icd10