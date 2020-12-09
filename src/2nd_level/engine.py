import tqdm
import torch
import numpy as np

import utils
import config


def optimizer_loss(gt_multimodal, gt_multimodal_conf, pred, pred_conf):
    gt_multimodal = gt_multimodal.reshape(-1, config.N_INPUT_MODES, 50, 2)
    error = torch.sum((pred[:, None, :, :, :] - gt_multimodal[:, :, None, :, :]) ** 2, axis=(-1, -2))
    log_weights = torch.log(pred_conf + 1e-8)[:, None, :]
    x = log_weights - 0.5 * error
    loss = -torch.logsumexp(x, axis=(2,))
    loss = loss * gt_multimodal_conf
    loss = loss.sum(dim=1)
    return loss.mean()


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()

    tk0 = tqdm.tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        inputs = d['inputs']
        input_confs = d['input_confs']

        inputs = inputs.to(device, dtype=torch.float)
        input_confs = input_confs.to(device, dtype=torch.float)

        model.zero_grad()
        pred, confidences = model(inputs, input_confs)

        loss = optimizer_loss(inputs, input_confs, pred, confidences)
        loss.backward()

        optimizer.step()

        if scheduler:
            scheduler.step()

        losses.update(loss.item(), inputs.size(0))
        tk0.set_postfix(loss=losses.avg)


def infer_fn(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()

    pred_coords = []
    pred_confs = []
    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            inputs = d['inputs']
            input_confs = d['input_confs']

            inputs = inputs.to(device, dtype=torch.float)
            input_confs = input_confs.to(device, dtype=torch.float)

            pred, confidences = model(inputs, input_confs)

            loss = optimizer_loss(inputs, input_confs, pred, confidences)

            pred = pred.cpu().detach().numpy()
            pred_coords.append(pred)
            confidences = confidences.cpu().detach().numpy()
            pred_confs.append(confidences)

            losses.update(loss.item(), inputs.size(0))
            tk0.set_postfix(loss=losses.avg)

    print(f'Loss = {losses.avg}')

    pred_coords = np.concatenate(pred_coords, axis=0)
    pred_confs = np.concatenate(pred_confs, axis=0)

    return pred_coords, pred_confs
