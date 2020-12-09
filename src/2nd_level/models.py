import torch
import torch.nn as nn

import config
from modules import ISAB, SAB, PMA


class SetTransformer(nn.Module):
    def __init__(self, dim_input=50 * 2 + 1, num_outputs=3, dim_output=50 * 2,
                 num_inds=8, dim_hidden=64, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.regressor = nn.Linear(dim_hidden, dim_output)
        self.mode_confidences = nn.Linear(dim_hidden, 1)

    def forward(self, X, input_confs):
        input_confs = torch.log(input_confs.unsqueeze(-1))
        X = torch.cat([X, input_confs], dim=-1)
        
        # Remove 16-mode from input
        X = X[:, :config.N_3_MODES, :]

        X = self.dec(self.enc(X))

        coords = self.regressor(X).reshape(-1, 3, 50, 2)
        confidences = torch.squeeze(self.mode_confidences(X), -1)
        confidences = torch.softmax(confidences, dim=1)

        return coords, confidences
