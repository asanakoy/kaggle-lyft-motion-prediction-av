import numpy as np
import torch

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
        shuffle=True)
    
    device = torch.device('cuda')
    model = models.SetTransformer(**config.MODEL_PARAMS)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, pct_start=config.PCT_START,
        div_factor=config.DIV_FACTOR, max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS, steps_per_epoch=len(data_loader))

    for epoch in range(config.EPOCHS):
        engine.train_fn(data_loader, model, optimizer,
                        device, scheduler=scheduler)
    
    torch.save(model.state_dict(), config.MODEL_PATH + 'transformer.bin') 


if __name__ == '__main__':
    run()
