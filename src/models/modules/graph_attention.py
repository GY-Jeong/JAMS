import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, num_classes_icd9, num_classes_icd10, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.U_9 = nn.Parameter(torch.empty(size=(num_classes_icd9, in_features)))
        self.U_10 = nn.Parameter(torch.empty(size=(num_classes_icd10, in_features)))
        
        # TODO : graph attention layer에서 사요ㅕㅇ하는 것처럼 atteentuion score 계산 : concat -> 1차원
        # TODO : 간선 정보 추가 외부에서 가져와야됨
        # TODO : adj matrix 정의
        self.adj_9to10 = torch.FloatTensor(np.load('/home/gy/workspace/AMC/medical-coding-reproducibility/code_convert/adj_matrix_icd9_to_icd10.npy'))
        self.adj_10to9 = torch.FloatTensor(np.load('/home/gy/workspace/AMC/medical-coding-reproducibility/code_convert/adj_matrix_icd10_to_icd9.npy'))

        ones_column = torch.ones((self.adj_9to10.shape[0], 1))
        self.adj_9to10 = torch.cat((ones_column, self.adj_9to10), dim=1).to('cuda')

        ones_column = torch.ones((self.adj_10to9.shape[0], 1))
        self.adj_10to9 = torch.cat((ones_column, self.adj_10to9), dim=1).to('cuda')

        # TODO : 따로? 같이? 실험 대상[임
        # self.W_9 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # self.W_10 = nn.Parameter(torch.empty(size=(in_features, out_features)))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self._init_weights(mean=0.0, std=0.03)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.U_9, mean, std)
        torch.nn.init.normal_(self.U_10, mean, std)
        # torch.nn.init.normal_(self.W, mean, std)
        # torch.nn.init.normal_(self.W_9, mean, std)
        # torch.nn.init.normal_(self.W_10, mean, std)

    def forward(self, version):
        if version == 'mimiciv_icd9':
            e = self._prepare_attentional_mechanism_input_sparse(self.U_9, self.U_10, self.adj_9to10[:, 1:])

            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(self.adj_9to10 > 0, e, zero_vec)
            
            attention = F.softmax(attention, dim=-1)
            attention = attention * self.adj_9to10
            attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)

            self_attention_score = attention[:, :1]
            cross_attention_score = attention[:, 1:]

            h_prime = self_attention_score * self.U_9 + torch.matmul(cross_attention_score, self.U_10)
        else:
            e = self._prepare_attentional_mechanism_input_sparse(self.U_10, self.U_9, self.adj_10to9[:, 1:])

            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(self.adj_10to9 > 0, e, zero_vec)

            attention = F.softmax(attention, dim=1)
            attention = attention * self.adj_10to9
            attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)

            self_attention_score = attention[:, :1]
            cross_attention_score = attention[:, 1:]

            h_prime = self_attention_score * self.U_10 + torch.matmul(cross_attention_score, self.U_9)
        return F.elu(h_prime)
    

    def _prepare_attentional_mechanism_input_sparse(self, query, key, adj):
        # Find the non-zero indices in the adjacency matrix
        non_zero_indices = adj.nonzero().t().contiguous()

        # Get the corresponding query and key vectors
        sparse_query = query[non_zero_indices[0]]
        sparse_key = key[non_zero_indices[1]]
        
        # Calculate the concatenated feature vectors for these non-zero indices
        concat_features = torch.cat((sparse_query, sparse_key), dim=1)

        # Calculate attention scores
        e = torch.matmul(concat_features, self.a).squeeze()
        e = self.leakyrelu(e)

        # Create a sparse tensor for the attention scores
        sparse_attention_scores = torch.sparse_coo_tensor(non_zero_indices, e, size=(query.size(0), key.size(0)))
        sparse_attn_e = sparse_attention_scores.to_dense()

        # 1. Expand query to have an extra dimension: (|Q|, 1, d)
        query_expanded = query.unsqueeze(1)  # shape becomes (|Q|, 1, d)

        # 2. Concatenate along the second dimension to make it (|Q|, 1, 2d)
        self_attn_feature = torch.cat((query_expanded, query_expanded), dim=2)  # shape becomes (|Q|, 1, 2d)

        self_attn_e = torch.matmul(self_attn_feature, self.a).squeeze(dim=-1)
        self_attn_e = self.leakyrelu(self_attn_e)

        attention_scores = torch.cat((self_attn_e, sparse_attn_e), dim=1)
        return attention_scores


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
