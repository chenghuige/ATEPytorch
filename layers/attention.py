import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, m_dim, score_func="sdp"):

        super(Attention, self).__init__()

        # m_dim: model dimension, dimension of query and key
        # score_func: sdp || mlp
        self.score_func = score_func

        if self.score_func == "sdp":
            self.scale_term = np.power(m_dim, 0.5)
        elif self.score_func == "mlp":
            self.mlp = nn.Linear(2 * m_dim, 1)

    def forward(self, *input):

        query, key, value = input
        # query: [batch_size, m_dim]
        # key: [batch_size, k_len, m_dim]
        # value: [batch_size, v_len, v_dim], v_len = k_len

        _, k_len, _ = key.shape

        if self.score_func == "sdp":
            # [batch_size, k_len] <= squeeze([batch_size, k_len, m_dim]*[batch_size, m_dim, 1], dim=2)
            score = torch.matmul(key, query.unsqueeze(dim=1).permute(0, 2, 1)).squeeze(dim=-1).div(self.scale_term) # [batch_size, k_len]
        elif self.score_func == "mlp":
            query = query.unsqueeze(dim=1).repeat(1, k_len, 1) # [batch_size, k_len, m_dim]
            cat_qk = torch.cat((query, key), dim=-1) # [batch_size, k_len, 2*m_dim]
            score = self.mlp(cat_qk).squeeze(dim=-1) # [batch_size, k_len]
        else:
            raise Exception("the given score function parameter doesn't exists...")

        score = F.softmax(score, dim=-1)  # [batch_size, k_len]
        # [batch_size, v_dim] <= squeeze([batch_size, 1, k_len] * [batch_size, v_len, v_dim], dim=1)
        output = torch.matmul(score.unsqueeze(dim=1), value).squeeze(dim=1)  # [batch_size, v_dim]

        return score, output