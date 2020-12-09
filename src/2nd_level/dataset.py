import torch
import numpy as np

import config


class Lyft2ndLevelDataset:
    def __init__(self, pred_paths):
        preds = [np.load(path) for path in pred_paths]
        self.inputs = np.concatenate(
            [pred['coords'] for pred in preds], axis=1).reshape(
                -1, config.N_INPUT_MODES, 50 * 2)
        self.input_confs = np.concatenate(
            [pred['confs'] for pred in preds], axis=1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {'inputs': torch.tensor(self.inputs[item], dtype=torch.float),
                'input_confs': torch.tensor(self.input_confs[item], dtype=torch.float)}
