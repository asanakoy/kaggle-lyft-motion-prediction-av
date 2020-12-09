import numpy as np
import pandas as pd
import torch
from l5kit.evaluation import write_pred_csv

import config
import dataset
import models
import engine
import utils


def run():
    utils.seed_everything(seed=config.SEED)

    train_dataset = dataset.Lyft2ndLevelDataset(
        config.PRED_PATHS + [config.MODE_16_PATH])
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=4,
        shuffle=False)
    
    device = torch.device('cuda')
    model = models.SetTransformer(**config.MODEL_PARAMS)
    model = model.to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH + 'transformer.bin'))

    coords, confs = engine.infer_fn(data_loader, model, device)

    sample_sub = pd.read_csv(config.SAMPLE_SUB_PATH)

    write_pred_csv(config.INFER_SAVE_PATH + 'submission.csv',
        timestamps=sample_sub['timestamp'],
        track_ids=sample_sub['track_id'],
        coords=coords,
        confs=confs)
 

if __name__ == '__main__':
    run()
